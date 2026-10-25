"""Trusted, bounded Actions checkpoints; caches are never the source of truth."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import stat
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
import zipfile

PREFIX = 'ionoscope-cloud-state-'
WORKFLOW = '.github/workflows/cloud-shadow.yml'
MAX_ARCHIVE = 64 * 1024 * 1024
MAX_UNPACKED = 512 * 1024 * 1024


class SafeRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != 'https':
            raise ValueError('Insecure redirect refused')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urlparse(req.full_url).netloc != urlparse(newurl).netloc:
            redirected.remove_header('Authorization')
        return redirected


def request(url):
    headers = {'User-Agent': 'ionoscope-cloud-shadow', 'Accept': 'application/vnd.github+json'}
    token = os.getenv('GH_TOKEN') or os.getenv('GITHUB_TOKEN')
    if token and urlparse(url).netloc == 'api.github.com':
        headers['Authorization'] = 'Bearer ' + token
    return build_opener(SafeRedirect()).open(Request(url, headers=headers), timeout=60)


def api(path, *, delete=False):
    url = 'https://api.github.com/repos/' + os.environ['GITHUB_REPOSITORY'] + path
    if delete:
        token = os.environ['GH_TOKEN']
        with build_opener(SafeRedirect()).open(Request(url, headers={
                'Authorization': 'Bearer ' + token, 'User-Agent': 'ionoscope-cloud-shadow'}, method='DELETE'), timeout=60):
            return
    with request(url) as response:
        return json.load(response)


def download(url, path, *, limit=MAX_ARCHIVE):
    temporary = path.with_suffix('.tmp')
    with request(url) as response, temporary.open('wb') as out:
        size = 0
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise ValueError('Download exceeds checkpoint size limit')
            out.write(chunk)
    temporary.replace(path)


def allowed(name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or '..' in path.parts or chr(92) in name or ':' in name:
        return False
    if len(path.parts) == 1:
        return name in {'cloud_run.json', 'collection.json'}
    if path.parts[0] == 'shadow':
        return name in {'shadow/shadow.sqlite3', 'shadow/status.json'}
    if path.parts[0] == 'rate_limit':
        return name == 'rate_limit/state.json'
    return path.parts[0] == 'history' and path.suffix in {'.txt', '.csv'}


def extract_checkpoint(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) > 1000 or sum(i.file_size for i in infos) > MAX_UNPACKED:
            raise ValueError('Checkpoint expansion limit exceeded')
        names = {i.filename for i in infos}
        if len(names) != len(infos):
            raise ValueError('Duplicate checkpoint members')
        for item in infos:
            if not allowed(item.filename) or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Unexpected checkpoint member: ' + item.filename)
        if not {'cloud_run.json', 'shadow/shadow.sqlite3'} <= names:
            raise ValueError('Incomplete checkpoint')
        for item in infos:
            target = destination / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, target.open('wb') as out:
                shutil.copyfileobj(source, out)
    with closing(sqlite3.connect(destination / 'shadow/shadow.sqlite3')) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Checkpoint database failed integrity check')


def trusted(artifact):
    run = api('/actions/runs/' + str(artifact['workflow_run']['id']))
    return (run['path'] == WORKFLOW and run['head_branch'] == 'main' and
            run['event'] in {'schedule', 'workflow_dispatch'} and
            run['head_repository']['full_name'] == os.environ['GITHUB_REPOSITORY'])


def restore(destination, initialize=False):
    page = api('/actions/artifacts?per_page=100')
    artifacts = sorted((a for a in page['artifacts'] if a['name'].startswith(PREFIX)),
                       key=lambda a: (a.get('created_at', ''), a['id']), reverse=True)
    for artifact in artifacts:
        if not trusted(artifact):
            continue
        if artifact['expired']:
            raise ValueError('Latest state expired. Recover a backup; do not silently restart the experiment.')
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'state.zip'
            download(artifact['archive_download_url'], archive)
            if artifact.get('digest'):
                actual = 'sha256:' + hashlib.sha256(archive.read_bytes()).hexdigest()
                if actual != artifact['digest']:
                    raise ValueError('Checkpoint digest mismatch')
            with zipfile.ZipFile(archive) as outer:
                if outer.namelist() != ['cloud-checkpoint.zip'] or outer.infolist()[0].file_size > MAX_ARCHIVE:
                    raise ValueError('Unexpected Actions artifact layout')
                inner = Path(temp) / 'checkpoint.zip'
                with outer.open('cloud-checkpoint.zip') as source, inner.open('wb') as out:
                    shutil.copyfileobj(source, out)
            extract_checkpoint(inner, destination)
        return artifact['id']
    if not initialize:
        raise ValueError('No checkpoint. First manual run must explicitly select initialize=true.')
    Path(destination).mkdir(parents=True, exist_ok=True)
    return None


def snapshot(state, archive):
    state, archive = Path(state), Path(archive)
    if not (state / 'shadow/shadow.sqlite3').exists():
        raise ValueError('No run database; refusing an empty checkpoint')
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive.with_suffix('.tmp')
    with tempfile.TemporaryDirectory() as temp:
        db_copy = Path(temp) / 'snapshot.sqlite3'
        with closing(sqlite3.connect(state / 'shadow/shadow.sqlite3')) as source, closing(sqlite3.connect(db_copy)) as target:
            source.backup(target)
        files = [(path, path.relative_to(state).as_posix()) for path in sorted(state.rglob('*'))
                 if path.is_file() and allowed(path.relative_to(state).as_posix()) and
                 path.relative_to(state).as_posix() != 'shadow/shadow.sqlite3']
        if len(files) + 1 > 1000 or db_copy.stat().st_size + sum(p.stat().st_size for p, _ in files) > MAX_UNPACKED:
            raise ValueError('State exceeds checkpoint expansion limit')
        with zipfile.ZipFile(temporary_archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path, name in files:
                bundle.write(path, name)
            bundle.write(db_copy, 'shadow/shadow.sqlite3')
    if temporary_archive.stat().st_size > MAX_ARCHIVE:
        temporary_archive.unlink()
        raise ValueError('Checkpoint exceeds 64 MiB; increase storage plan before continuing')
    temporary_archive.replace(archive)


def cleanup(keep=3):
    page = api('/actions/artifacts?per_page=100')
    artifacts = sorted((a for a in page['artifacts'] if a['name'].startswith(PREFIX)),
                       key=lambda a: (a.get('created_at', ''), a['id']), reverse=True)
    artifacts = [a for a in artifacts if trusted(a)]
    for artifact in artifacts[keep:]:
        api('/actions/artifacts/' + str(artifact['id']), delete=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['restore', 'snapshot', 'cleanup'])
    parser.add_argument('--state', default='cloud-state')
    parser.add_argument('--archive', default='cloud-checkpoint.zip')
    parser.add_argument('--initialize', action='store_true')
    args = parser.parse_args()
    if args.command == 'restore':
        print(json.dumps({'restored_artifact': restore(args.state, args.initialize)}))
    elif args.command == 'snapshot':
        snapshot(args.state, args.archive)
    else:
        cleanup()


if __name__ == '__main__':
    main()

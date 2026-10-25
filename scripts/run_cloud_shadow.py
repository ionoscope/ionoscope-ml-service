"""One prospective collection/inference cycle, without a resident API or Docker."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cloud_artifacts import download
from app.services.shadow import ShadowRun, digest


def compact_history(merged, destination, cutoff):
    """Only the last nine days of canonical source data survive the next job."""
    import pandas as pd
    for path in (merged / 'raw/giro').glob('station=*/history.txt'):
        target = destination / path.relative_to(merged)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for line in path.read_text().splitlines():
            try:
                stamp = datetime.fromisoformat(line.split()[0].replace('Z', '+00:00'))
                if stamp.tzinfo and stamp >= cutoff:
                    lines.append(line)
            except (ValueError, IndexError):
                continue
        target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for path in (merged / 'processed').glob('*.csv'):
        if path.stat().st_size < 3:
            continue
        frame = pd.read_csv(path)
        if 'time_utc' not in frame:
            continue
        times = pd.to_datetime(frame.time_utc, utc=True, errors='coerce', format='mixed')
        target = destination / 'processed' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.loc[times >= cutoff].to_csv(target, index=False)


def collect(config, directory, start, end):
    from collect_hf_data import init_paths, collect_giro, collect_gfz_indices, collect_omni, write_csv
    paths = init_paths(config, argparse.Namespace(output_dir=str(directory)))
    errors = []

    def gfz():
        rows = []
        for index in config['gfz']['indices']:
            try:
                individual = {**config, 'gfz': {**config['gfz'], 'indices': [index]}}
                rows.extend(collect_gfz_indices(individual, paths, start, end, 20))
            except Exception as exc:
                errors.append({'source': 'gfz', 'index': index, 'error': type(exc).__name__})
        write_csv(paths.processed / 'geophysical_indices.csv', rows)

    def omni():
        try:
            write_csv(paths.processed / 'omni_solar_wind.csv', collect_omni(config, paths, start, end, 20))
        except Exception as exc:
            errors.append({'source': 'omni', 'error': type(exc).__name__})

    def giro():
        try:
            rows = collect_giro(config, paths, start, end, 20)
            errors.extend({'source': 'giro', 'station': row['station'], 'error': row['error']}
                          for row in rows if row.get('error'))
        except Exception as exc:
            errors.append({'source': 'giro', 'error': type(exc).__name__})

    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in [pool.submit(gfz), pool.submit(omni), pool.submit(giro)]:
            future.result()
    return errors


def models(plan, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for spec in plan['models'].values():
        path = directory / spec['file']
        if not path.exists():
            repository = os.environ['GITHUB_REPOSITORY']
            download(f'https://github.com/{repository}/releases/download/{plan["release_tag"]}/{spec["file"]}', path, limit=8 * 1024 * 1024)
        if digest(path) != spec['sha256']:
            raise ValueError('Frozen model SHA-256 mismatch: ' + spec['file'])


def run_cycle(plan_path, state, data_root, model_dir, *, clock=None, collector=None):
    clock = clock or (lambda: datetime.now(UTC))
    plan = json.loads(Path(plan_path).read_text())
    state, data_root, model_dir = Path(state).resolve(), Path(data_root).resolve(), Path(model_dir).resolve()
    actual_ref = subprocess.check_output(['git', '-C', str(data_root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_ref != plan['data_commit']:
        raise ValueError('Data checkout differs from the frozen cloud plan')
    sys.path.insert(0, str(data_root))
    from incremental_update import merge_batches
    from data_preparation.candidate_inputs import prepare_live, PROFILE

    models(plan, model_dir)  # Validate hashes before unpickling.
    identity = hashlib.sha256(Path(plan_path).read_bytes()).hexdigest()
    ml_root = Path(__file__).resolve().parents[1]
    implementation = hashlib.sha256(b''.join((ml_root / name).read_bytes() for name in [
        'scripts/run_cloud_shadow.py', 'app/services/shadow.py',
        'app/services/prediction_service.py', 'app/services/model_loader.py',
        'app/schemas/prediction.py', 'requirements.txt'])).hexdigest()
    manifest_path = state / 'cloud_run.json'
    state.mkdir(parents=True, exist_ok=True)
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and previous['plan_sha256'] != identity:
        raise ValueError('Cloud plan changed: explicitly start a separate experiment')
    if previous and previous.get('ml_implementation_sha256') != implementation:
        raise ValueError('Inference implementation changed: use a separate experiment')
    end = clock().replace(microsecond=0)
    manifest = previous or {'plan_sha256': identity, 'ml_implementation_sha256': implementation,
                           'started_at': end.isoformat(), 'environment': 'github-actions', 'last_collection_end': None}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    run = ShadowRun(state / 'shadow', model_dir / plan['models']['original']['file'],
                    model_dir / plan['models']['candidate']['file'], plan['models']['candidate']['sha256'],
                    plan['stations'], now=end)
    try:
        start = end - timedelta(days=8)
        if manifest.get('last_collection_end'):
            start = max(start, datetime.fromisoformat(manifest['last_collection_end']) - timedelta(hours=3))
        config = tomllib.loads((data_root / 'config.toml').read_text())
        config['giro'].update(stations=plan['stations'], workers=2)
        os.environ['IONOSCOPE_GIRO_RATE_DIR'] = str(state / 'rate_limit')
        with tempfile.TemporaryDirectory(prefix='ionoscope-cloud-') as temp:
            temp = Path(temp)
            generation_name = 'generation-' + end.strftime('%Y%m%dT%H%M%S')
            batch, merged, generation = temp / 'batch', temp / 'merged', temp / generation_name
            errors = (collector or collect)(config, batch, start, end)
            merge_batches([state / 'history', batch], merged)
            compact_history(merged, state / 'history', end - timedelta(days=9))
            manifest['last_collection_end'] = end.isoformat()
            manifest_path.write_text(json.dumps(manifest, indent=2))
            (state / 'collection.json').write_text(json.dumps({'start': start.isoformat(), 'end': end.isoformat(), 'errors': errors}, indent=2))
            prepare_live(state / 'history', generation / 'normalized', generation / 'features', plan['stations'], end)
            now = clock()
            generation_manifest = {'generation': generation_name, 'input_profile': PROFILE,
                                   'prepared_at': now.isoformat(), 'stations': plan['stations'],
                                   'available_stations': [p.stem.removesuffix('_features') for p in
                                                          (generation / 'features/stations').glob('*_features.csv')]}
            (temp / 'state.json').write_text(json.dumps(generation_manifest))
            report = run.cycle(temp, now=now, clock=clock)
        return report
    finally:
        run.close()


def summary(report):
    lines = ['## Ionoscope: prospective cloud comparison', '',
             '**Experimental scheduled collection; the production API is unchanged.**', '',
             f"Phase: {report['phase']}; common scored pairs: {report['common_scored']}; days with scored pairs: {report['common_days']}.", '',
             '| Method | MAE (MHz), common pairs | Coverage |', '|---|---:|---:|']
    for name in ['original', 'candidate', 'persistence']:
        mae = report[name]['mae']
        coverage = report['coverage'].get(name)
        lines.append(f"| {name} | {mae if mae is not None else 'no facts yet'} | {coverage if coverage is not None else '—'} |")
    lines += ['', 'Delayed or missed schedules remain missing; no retrospective predictions.',
              'A small sample or low coverage does not demonstrate a deployable improvement.',
              'Checkpoint preserves source history and the immutable prediction journal. No automatic promotion.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(Path(__file__).resolve().parents[1] / 'configs/cloud-shadow.json'))
    parser.add_argument('--state', default='cloud-state')
    parser.add_argument('--data-root', default='../ionoscope-data')
    parser.add_argument('--models', default='cloud-models')
    args = parser.parse_args()
    report = run_cycle(args.plan, args.state, args.data_root, args.models)
    text = summary(report)
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as out:
            out.write(text)
    print(text)


if __name__ == '__main__':
    main()

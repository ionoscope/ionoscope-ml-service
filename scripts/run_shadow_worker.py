"""Continuously compare frozen models on the exact published live generation."""
import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.model_loader import DEFAULT_MODEL_PATH
from app.services.shadow import ShadowRun, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default='/app/live_data')
    parser.add_argument('--output', default='/app/shadow_data')
    parser.add_argument('--candidate', default=os.getenv('IONOSCOPE_SHADOW_MODEL_PATH'))
    parser.add_argument('--candidate-sha256', default=os.getenv('IONOSCOPE_SHADOW_MODEL_SHA256'))
    parser.add_argument('--original-sha256', default='c32e2157a7d066ff352c61db44ec48d33888b76fd270eb1520ef4cb708625b82')
    parser.add_argument('--stations', default='AL945,AT138,CAJ2M,DB049,EB040,GA762,GM037,GR13L,IF843,JI91J,JR055,NI135,PQ052,RO041,SO148,TR169,WA619')
    parser.add_argument('--interval-seconds', type=float, default=30)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if not args.candidate or not args.candidate_sha256 or args.interval_seconds <= 0:
        parser.error('Pinned candidate path/hash and positive interval required')
    if digest(DEFAULT_MODEL_PATH) != args.original_sha256:
        parser.error('Served model differs from the frozen original; use a new explicit evaluation plan')
    run = ShadowRun(args.output, DEFAULT_MODEL_PATH, args.candidate, args.candidate_sha256,
                    [s.strip().upper() for s in args.stations.split(',') if s.strip()])
    try:
        while True:
            result = run.cycle(args.workspace)
            print(json.dumps({k: result[k] for k in ['updated_at', 'opportunities', 'common_scored', 'error']}), flush=True)
            if args.once:
                return
            time.sleep(args.interval_seconds)
    finally:
        run.close()


if __name__ == '__main__':
    main()

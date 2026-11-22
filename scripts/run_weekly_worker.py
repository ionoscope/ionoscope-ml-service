"""Weekly retraining, daily seven-day forecasts and prospective scoring."""
import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.weekly import WeeklyExperiment, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', default='/app/live_data')
    parser.add_argument('--output', default='/app/weekly_data')
    parser.add_argument('--interval-seconds', type=float, default=300)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        parser.error('Positive interval required')
    run = WeeklyExperiment(args.output)
    try:
        while True:
            try:
                status = run.cycle(args.workspace)
            except Exception as exc:
                status = {'phase':'failed','updated_at':datetime.now(UTC).isoformat(),
                          'error':str(exc),'automatic_promotion':False}
            atomic_json(Path(args.output)/'status.json',status)
            print(json.dumps(status),flush=True)
            if args.once:
                if status['phase'] != 'ok':
                    raise SystemExit(1)
                return
            time.sleep(args.interval_seconds)
    finally:
        run.close()


if __name__ == '__main__':
    main()

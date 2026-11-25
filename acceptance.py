"""Run both predictions inside the delivery and verify them against the archived submissions."""
import argparse
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TASKS = ('rank_prediction',)


def audited_stage(task, stage, trained):
    project = ROOT / task
    forbidden = {project / 'submission'}
    if stage == 'predict':
        forbidden.add(project / 'outputs' / ('submission_player.csv' if task == 'player_identification' else 'submission_rank.csv'))

    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(ROOT.parent) and not path.is_relative_to(ROOT):
            raise PermissionError(f'Attempted access outside the delivered project: {path}')
        if stage == 'predict' and any(path.is_relative_to(f) for f in forbidden):
            raise PermissionError(f'Prediction attempted to open a reference or existing output: {path}')

    sys.addaudithook(audit)
    script = project / 'scripts' / f'{stage}.py'
    sys.argv = [str(script)] + (['--trained'] if stage == 'predict' and trained else [])
    runpy.run_path(str(script), run_name='__main__')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trained', action='store_true', help='Validate locally retrained weights')
    parser.add_argument('--stage', choices=('predict', 'verify'), help=argparse.SUPPRESS)
    parser.add_argument('--task', choices=TASKS)
    args = parser.parse_args()
    if args.stage:
        audited_stage(args.task, args.stage, args.trained)
        return
    for task in ([args.task] if args.task else TASKS):
        for stage in ('predict', 'verify'):
            cmd = [sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--task', task, '--stage', stage]
            if args.trained:
                cmd.append('--trained')
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()

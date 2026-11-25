from pathlib import Path
import sys

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK.parent))
from verify_submission import verify


def main():
    verify(TASK, 'submission_rank.csv')

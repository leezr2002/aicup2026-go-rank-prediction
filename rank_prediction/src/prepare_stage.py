"""Build model inputs from branch estimates and the official SGF question metadata."""
import csv
import json
import re
from pathlib import Path
from graph import logp, np, sample_input, softmax

TASK = Path(__file__).resolve().parents[1]
DATA = TASK.parent / 'data/official/rank_prediction'
CONFIG = json.loads((TASK / 'config/final.yaml').read_text(encoding='utf-8'))
BRANCHES = TASK / 'artifacts/branches'
OUTPUT = TASK / 'outputs'


def design(table, sizes):
    history = table['history']
    hmm = softmax(logp(history) + table['policy_delta'], axis=1)
    return sample_input(history, table['opponent'], table['fingerprint'], table['crosscolor'],
                        table['kernel'], hmm, sizes)


def training_inputs():
    with np.load(BRANCHES / 'training.npz') as table:
        x, anchor = design(table, table['sizes'])
        return x, anchor, table['rank'], table['player'], table['fold']


def check_official_files(directory, config):
    from hashlib import file_digest
    for name, expected in config['official_inputs'].items():
        with (directory / name).open('rb') as handle:
            if file_digest(handle, 'sha256').hexdigest() != expected:
                raise ValueError(f'Official input differs from the frozen branch inputs: {name}')


def official_inputs():
    check_official_files(DATA, CONFIG)
    ids, counts = [], []
    for side in ('public', 'private'):
        with (DATA / f'rank_prediction_test_{side}.csv').open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                count = int(row['num_games'])
                for i in range(1, count + 1):
                    sgf = row[f'sgf_{i}']
                    hd = re.search(r'(?:HD|HA)\[(\d+)\]', sgf)
                    if not sgf or (hd and int(hd[1]) > 1):
                        raise ValueError(f"Unsupported game in {row['question_id']}")
                ids.append(row['question_id'])
                counts.append(count)
    with np.load(BRANCHES / 'official.npz') as table:
        if list(table['question_ids']) != ids:
            raise ValueError('Branch question order differs from the official input')
        counts = np.array(counts)
        x, anchor = design(table, counts)
        return dict(ids=np.array(ids), counts=counts, x=x, anchor=anchor, cross=table['crosscolor_anchored'])


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / 'official_inputs.npz', **official_inputs())
    print('Prepared 800 official questions')

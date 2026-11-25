"""Validate official IDs, task-specific values and exact CSV serialization."""
import csv
from hashlib import sha256
import json
from pathlib import Path


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.reader(handle))


def verify(project, filename):
    root = project.parent
    player = filename == 'submission_player.csv'
    config = json.loads((project / 'config/final.yaml').read_text(encoding='utf-8'))
    data = root / 'data/official' / project.name
    template = data / ('sample_submission_player_v2.csv' if player else 'sample_submission_rank.csv')
    ref = project / 'submission' / filename
    output = project / 'outputs' / filename
    digest = sha256(ref.read_bytes()).hexdigest()
    if digest != config['sha256']:
        raise ValueError('Archived submission checksum mismatch')
    rows, sample, expected = read(output), read(template), read(ref)
    count = 1600 if player else 800
    if len(rows) != count + 1 or rows[0] != sample[0]:
        raise ValueError('Output row count or header differs from the template')
    ids = [row[0] for row in rows[1:]]
    if len(set(ids)) != count or ids != [row[0] for row in sample[1:]]:
        raise ValueError('Question IDs differ from the template')
    if player:
        candidates = {}
        for key, suffix, side in [('pub', '', 'public'), ('priv', '', 'private'),
                                  ('pub2', '2', 'public'), ('priv2', '2', 'private')]:
            path = data / f'player_identification_candidates{suffix}_{side}.csv'
            with path.open(encoding='utf-8-sig', newline='') as handle:
                candidates[key] = {r['player_id'] for r in csv.DictReader(handle)}
        for qid, *values in rows[1:]:
            if len(values) != 5 or len(set(values)) != 5 or not set(values) <= candidates[qid.split('_')[0]]:
                raise ValueError(f'Invalid Top-5 for {qid}')
    elif any(len(row) != 2 or row[1] not in config['ranks'] for row in rows[1:]):
        raise ValueError('Invalid rank')
    match = sum(x == y for x, y in zip(rows[1:], expected[1:], strict=True))
    output_hash = sha256(output.read_bytes()).hexdigest()
    if match != count or digest != output_hash:
        raise ValueError(f'{match}/{count} rows match; output SHA-256 {output_hash}, expected {digest}')
    report = {'file': filename, 'questions': count, 'matching_rows': match, 'sha256': output_hash}
    (project / 'outputs/verification.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'{filename}: {match}/{count} rows and SHA-256 match ({output_hash})')

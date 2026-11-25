"""Apply the isolated-neighbor graph separately to Public and Private."""
import argparse
import csv
from threadpoolctl import threadpool_limits
from graph import Margin, edges, factors, logp, message, np, softmax, torch
from prepare_stage import CONFIG, TASK, OUTPUT, official_inputs


def predict(x, anchor, cross, counts, models):
    neighbors, distances = edges(x, anchor)
    evidence = factors(x, neighbors, models)
    delta = message(anchor, neighbors, distances, evidence, CONFIG['neighbors'])
    graph = softmax(logp(anchor) + CONFIG['strength'] * delta, axis=1)
    weight = np.interp(counts, CONFIG['bins'], CONFIG['count'])[:, None]
    p = softmax(weight * logp(graph) + (1 - weight) * logp(cross), axis=1)
    utility = np.eye(10) + (np.eye(10, k=1) + np.eye(10, k=-1)) / np.e
    return np.array(CONFIG['ranks'])[(p @ utility).argmax(axis=1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained', action='store_true', help='Use weights produced by train.py')
    args = parser.parse_args()
    folder = TASK / ('outputs/weights' if args.trained else 'artifacts/weights')
    torch.set_num_threads(1)
    with threadpool_limits(1):
        data = official_inputs()
        models = []
        for fold in range(CONFIG['folds']):
            model = Margin(data['x'].shape[1])
            model.load_state_dict(torch.load(folder / f'fold{fold}.pt', map_location='cpu', weights_only=True))
            models.append(model.eval())
        ranks = []
        for prefix in ('pub_', 'priv_'):
            idx = np.char.startswith(data['ids'], prefix)
            if idx.sum() != 400:
                raise ValueError(f'Unexpected number of questions for {prefix}')
            ranks.extend(predict(data['x'][idx], data['anchor'][idx], data['cross'][idx],
                                 data['counts'][idx], models))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dest = OUTPUT / 'submission_rank.csv'
    temp = dest.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['question_id', 'pred_rank'])
        writer.writerows(zip(data['ids'], ranks, strict=True))
    temp.replace(dest)
    print(f'Wrote {len(ranks)} predictions to outputs/{dest.name}')

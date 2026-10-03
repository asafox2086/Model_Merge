"""Evaluate hard voting only; this is not the full Xie et al. TMI meta-ensemble."""

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from methods.pscore import client_probabilities
from scripts.summarize_pscore_baseline import DATASETS, MODELS, key, require, write_csv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT / 'outputs/pscore_full_20261003')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/voting_feasibility_20261003')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'docs/experiments/voting_feasibility')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    baseline_path = ROOT / 'My_merge_ret/reports/head_only_baselines_20260724_v5.csv'
    with baseline_path.open() as handle:
        lamp = {key(row): row for row in csv.DictReader(handle)
                if row['method'] == 'lamp_merge' and row['status'] == 'OK'}
    records, audit = [], []
    for feature_path in sorted(args.input_root.rglob('provenance.json')):
        provenance = json.loads(feature_path.read_text())
        dataset, model = feature_path.parent.parent.name, feature_path.parent.name
        test_path = feature_path.parent / 'test.pt'
        cached = torch.load(test_path, map_location='cpu', weights_only=False)
        labels = cached['labels'].numpy()
        source_root = feature_path.parents[3]
        cases = sorted((source_root / 'cases/small' / dataset / model).rglob('fusion.pt'))
        require(len(cases) == 9, 'Expected nine expert banks per dataset/backbone')
        for checkpoint_path in cases:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            original = json.loads((checkpoint_path.parent / 'result.json').read_text())
            protocol = json.loads((checkpoint_path.parent / 'protocol.json').read_text())
            require(protocol['features'] == provenance, 'Feature provenance mismatch')
            weights, biases = checkpoint['head_weights'], checkpoint['head_biases']
            for weight, bias, source in zip(weights, biases, protocol['heads']):
                require(hashlib.sha256(weight.numpy().tobytes() + bias.numpy().tobytes()).hexdigest()
                        == source['head_sha256'], 'Expert head hash mismatch')
            classes = weights.shape[1]
            probabilities = client_probabilities(cached['features'], weights, biases, device)
            expert_predictions = probabilities.reshape(len(labels), len(weights), classes).argmax(2).numpy()
            votes = (expert_predictions[:, :, None] == np.arange(classes)).sum(1)
            predicted = votes.argmax(1)
            tied = (votes == votes.max(1, keepdims=True)).sum(1) > 1
            mean_probabilities = probabilities.reshape(len(labels), len(weights), classes).mean(1).numpy()
            probability_tiebreak = np.where(votes == votes.max(1, keepdims=True), mean_probabilities, -1).argmax(1)
            sensitivity_confusion = np.bincount(labels * classes + probability_tiebreak,
                                               minlength=classes ** 2).reshape(classes, classes)
            sensitivity_denominator = sensitivity_confusion.sum(0) + sensitivity_confusion.sum(1)
            sensitivity_f1 = float(np.divide(2 * np.diag(sensitivity_confusion), sensitivity_denominator,
                                            out=np.zeros(classes), where=sensitivity_denominator != 0).mean())
            confusion = np.bincount(labels * classes + predicted, minlength=classes ** 2).reshape(classes, classes)
            np.testing.assert_array_equal(confusion.sum(1), np.fromstring(lamp[key(original)]['true_counts'], sep=' ', dtype=int))
            with np.load(checkpoint_path.parent / 'test_predictions.npz') as previous:
                np.testing.assert_array_equal(labels, previous['labels'])
            accuracy = float(np.trace(confusion) / confusion.sum())
            denominator = confusion.sum(0) + confusion.sum(1)
            f1 = float(np.divide(2 * np.diag(confusion), denominator,
                                out=np.zeros(classes), where=denominator != 0).mean())
            output = args.output_root / dataset / model / checkpoint_path.parent.relative_to(checkpoint_path.parents[3])
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output / 'predictions.npz', labels=labels, expert_predictions=expert_predictions,
                                votes=votes, predictions=predicted, mean_probabilities=mean_probabilities,
                                probability_tiebreak=probability_tiebreak)
            with np.load(output / 'predictions.npz') as saved:
                np.testing.assert_array_equal(saved['votes'].argmax(1), predicted)
            row = {field: original[field] for field in ('dataset', 'model', 'num_clients', 'beta', 'seed')}
            row.update(method='hard_vote_component_only', accuracy=accuracy, macro_f1=f1,
                       probability_tiebreak_accuracy=float(np.trace(sensitivity_confusion) / len(labels)),
                       probability_tiebreak_macro_f1=sensitivity_f1,
                       lamp_accuracy=float(lamp[key(original)]['accuracy']),
                       lamp_macro_f1=float(lamp[key(original)]['macro_f1']),
                       tie_samples=int(tied.sum()), num_samples=len(labels),
                       confusion_matrix=json.dumps(confusion.tolist()))
            records.append(row)
            audit.append({'case': str(checkpoint_path.parent), 'checkpoint_sha256': digest(checkpoint_path),
                          'features_sha256': digest(test_path), 'protocol_sha256': digest(checkpoint_path.parent / 'protocol.json'),
                          'predictions_path': str(output / 'predictions.npz'),
                          'predictions_sha256': digest(output / 'predictions.npz')})
        print(f'Completed {dataset} {model}: {len(records)}/180', flush=True)
    expected = set(itertools.product(DATASETS, MODELS, (3, 5, 7), (0.0, 0.01, 0.1), (42,)))
    require(len(records) == 180 and {key(row) for row in records} == expected, 'Incomplete or duplicate grid')
    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.report_dir / 'raw.csv', records)
    summary = []
    for model, dataset in itertools.product(MODELS, DATASETS):
        selected = [row for row in records if row['model'] == model and row['dataset'] == dataset]
        summary.append({'model': model, 'dataset': dataset,
                        **{metric: 100 * mean(row[metric] for row in selected)
                           for metric in ('accuracy', 'macro_f1', 'probability_tiebreak_accuracy',
                                          'probability_tiebreak_macro_f1', 'lamp_accuracy', 'lamp_macro_f1')}})
    write_csv(args.report_dir / 'summary.csv', summary)
    totals = {metric: 100 * mean(row[metric] for row in records)
              for metric in ('accuracy', 'macro_f1', 'probability_tiebreak_accuracy',
                             'probability_tiebreak_macro_f1', 'lamp_accuracy', 'lamp_macro_f1')}
    totals.update(cases=180, accuracy_wins=sum(row['accuracy'] > row['lamp_accuracy'] for row in records),
                  macro_f1_wins=sum(row['macro_f1'] > row['lamp_macro_f1'] for row in records))
    (args.report_dir / 'audit.json').write_text(json.dumps({
        'scope': 'Hard-voting component feasibility only, NOT the Xie meta-ensemble',
        'tie_rule': 'smallest class index; fixed before evaluation',
        'sensitivity_rule': 'maximum mean expert probability among tied classes; added after observing frequent ties; both reported',
        'source_sha256': digest(Path(__file__)), 'probability_code_sha256': digest(ROOT / 'methods/pscore.py'),
        'baseline_sha256': digest(baseline_path), 'totals': totals, 'cases': audit}, indent=2) + '\n')
    print(json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()

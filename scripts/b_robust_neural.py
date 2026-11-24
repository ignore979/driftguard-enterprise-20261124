"""Fixed-recipe B continuation study. Model updates consume only official training labels."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter, process_time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for variable in ('BASELINE_CPU_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from zhixun.bearing_study import bearing_gate, bearing_metrics, load_bearings
from zhixun.data import digest, verify_inputs, verify_parent, verify_split, write_json
from zhixun.model_extension import expand_hidden
from zhixun.vendor.model import Model


# Columns: arm, width, epochs, starting LR, device mass, own RUL bin mass,
# robust error scale (in original minutes), adaptive worst-device sampling mass.
RECIPES = (
    ('neural-24-mse', 24, 120, .003, 0., 0., 0., 0.),
    ('neural-24-device', 24, 120, .003, .5, 0., 0., 0.),
    ('neural-96-mse', 96, 160, .003, 0., 0., 0., 0.),
    ('neural-96-device', 96, 160, .003, .5, 0., 0., 0.),
    ('neural-96-device-stage', 96, 160, .003, .35, .25, 0., 0.),
    ('neural-96-smooth-huber', 96, 160, .003, .5, 0., 400., 0.),
    ('neural-128-device-stage', 128, 160, .003, .35, .25, 0., 0.),
    ('neural-96-adaptive-device', 96, 160, .003, .1, .1, 0., .4),
)
FIELDS = ('arm', 'width', 'epochs', 'initial_lr', 'device_mass', 'stage_mass',
          'smooth_huber_delta_minutes', 'adaptive_device_mass')
TAIL_RECIPES = (
    ('neural-96-late-overestimate4', 96, 160, .003, .35, .25, 0., 0.),
    ('neural-96-late-overestimate12', 96, 160, .003, .35, .25, 0., 0.),
)
TAIL_PENALTIES = {TAIL_RECIPES[0][0]: 4, TAIL_RECIPES[1][0]: 12}


def mean_one_group_weight(ids):
    _, indices, counts = np.unique(ids, return_inverse=True, return_counts=True)
    return indices, counts, len(indices) / (len(counts) * counts[indices])


def equal_stage_weights(device_index, device_counts, remaining):
    """Divide each device's observed life into three train-only relative RUL bins."""
    maximum = np.zeros(len(device_counts), dtype=np.float64)
    np.maximum.at(maximum, device_index, remaining)
    stage = np.minimum((3 * remaining / np.maximum(1, maximum[device_index])).astype(int), 2)
    ids = 3 * device_index + stage
    cells = np.bincount(ids, minlength=3 * len(device_counts)).reshape(-1, 3)
    stages_per_device = np.count_nonzero(cells, axis=1)
    return len(remaining) / (len(device_counts) * stages_per_device[device_index] *
                             cells[device_index, stage])


def objective(error, delta):
    if delta == 0:
        return error ** 2, 2 * error
    root = np.sqrt(1 + (error / delta) ** 2)
    return 2 * delta ** 2 * (root - 1), 2 * error / root


def fit_study(model, x, raw_target, groups, recipe, *, tail_penalty=0.):
    name, _, epochs, rate, device_mass, stage_mass, robust_minutes, adaptive_mass = recipe
    del name
    x = model.normalized(x)
    scale = model.meta['target_scale']
    y = (raw_target - model.meta['target_mean']) / scale
    delta = robust_minutes / scale
    group_index, counts, uniform_device = mean_one_group_weight(groups)
    equal_stage = equal_stage_weights(group_index, counts, raw_target)
    weight = 1 - device_mass - stage_mass - adaptive_mass
    weight = weight + device_mass * uniform_device + stage_mass * equal_stage
    group_logits = np.zeros(len(counts))
    near_failure = np.clip(1 - raw_target / 200, 0, 1)
    traces = []
    size = model.meta['batch_size']
    for i in range(epochs):
        # Timer/schedule are fixed before evaluating any validation run.
        # The final LR is one tenth of the starting LR.
        model.meta['learning_rate'] = rate * (.55 + .45 * np.cos(np.pi * i / max(1, epochs - 1)))
        current_weights = weight
        if adaptive_mass:
            probabilities = np.exp(group_logits - group_logits.max())
            probabilities /= probabilities.sum()
            probabilities = .9 * probabilities + .1 / len(counts)
            current_weights = weight + adaptive_mass * len(y) * probabilities[group_index] / counts[group_index]
        if not np.isclose(current_weights.mean(), 1, rtol=0, atol=1e-12):
            raise ValueError('Training weight mass changed')
        rng = np.random.default_rng(np.random.SeedSequence([model.meta['seed'], model.meta['epochs']]))
        order = rng.permutation(len(y))
        loss_sum = 0.
        for start in range(0, len(order), size):
            indices = order[start:start + size]
            batch = x[indices]
            hidden, logits = model.forward(batch)
            error = logits[:, 0] - y[indices]
            loss, gradient = objective(error, delta)
            importance = current_weights[indices]
            if tail_penalty:
                importance = importance * (1 + tail_penalty * near_failure[indices] * (error > 0))
            loss_sum += float(np.dot(loss, importance))
            gradient = (gradient * importance / len(indices))[:, None]
            previous = (gradient @ model.params['w2'].T) * (1 - hidden ** 2)
            gradients = {'w2': hidden.T @ gradient, 'b2': gradient.sum(axis=0),
                         'w1': batch.T @ previous, 'b1': previous.sum(axis=0)}
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
            if not math.isfinite(norm):
                raise ValueError('Nonfinite study gradient')
            model.meta['steps'] += 1
            t = model.meta['steps']
            for key, value in gradients.items():
                value *= min(1., 5. / max(norm, 1e-12))
                model.first[key] = .9 * model.first[key] + .1 * value
                model.second[key] = .999 * model.second[key] + .001 * value ** 2
                corrected_first = model.first[key] / (1 - .9 ** t)
                corrected_second = model.second[key] / (1 - .999 ** t)
                model.params[key] -= model.meta['learning_rate'] * corrected_first / (np.sqrt(corrected_second) + 1e-8)
        model.meta['epochs'] += 1
        model.validate()
        traces.append({'epoch': i + 1, 'train/objective': loss_sum / len(y),
                       'train/learning_rate': model.meta['learning_rate']})
        if adaptive_mass:
            error = model.forward(x)[1][:, 0] - y
            losses = np.bincount(group_index, weights=error ** 2) / counts
            group_logits = np.clip(group_logits + .10 * (losses - losses.mean()), -4, 4)
    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('initial', 'tail'), default='initial')
    suite = parser.parse_args().suite
    recipes = TAIL_RECIPES if suite == 'tail' else RECIPES
    verify_inputs('B', ROOT)
    source = ROOT / 'data/B/checkpoint.npz'
    train_path = ROOT / 'data/B/train.jsonl'
    validation_path = ROOT / 'data/B/validation.jsonl'
    baseline_dir = ROOT / 'artifacts/b-phase1-20261123T142953-3f0ddb42'
    candidate_study = json.loads((baseline_dir / 'local_candidate.json').read_text(encoding='utf8'))
    candidate_path = baseline_dir / candidate_study['candidate']
    if digest(candidate_path) != candidate_study['checkpoint_sha256']:
        raise ValueError('B2 reference identity mismatch')
    original = Model.load(source)
    train, groups = load_bearings(train_path, original)
    val, val_groups = load_bearings(validation_path, original)
    verify_parent(original, 'B', train)
    verify_split(train, val)
    before = bearing_metrics(Model.load(candidate_path).predict(val.x), val.y, val_groups)
    untrained = bearing_metrics(original.predict(val.x), val.y, val_groups)
    prefix = 'b-robust-neural-tail' if suite == 'tail' else 'b-robust-neural'
    name = f'{prefix}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}'
    folder = ROOT / 'artifacts' / name
    folder.mkdir(parents=True, exist_ok=False)
    registration = {
        'development_split': 'official_public_B_validation',
        'source_commit': '271263719f06c6b684a30c4d5421035b06e93369',
        'runner_sha256': digest(Path(__file__)),
        'source_sha256': digest(source), 'b2_checkpoint_sha256': digest(candidate_path),
        'train_sha256': digest(train_path), 'validation_sha256': digest(validation_path),
        'train_counts': dict(Counter(groups.tolist())), 'validation_counts': dict(Counter(val_groups.tolist())),
        'validation_use': 'final scores of predeclared arms only; one exploratory panel, no hidden data',
        'learning_rate_schedule': 'cosine floor .1 times the initial LR',
        'recipes': [dict(zip(FIELDS, recipe)) | {
            'near_failure_overestimate_penalty': TAIL_PENALTIES.get(recipe[0], 0),
        } for recipe in recipes],
        'b0': untrained, 'b2': before,
    }
    write_json(folder / 'registration.json', registration)
    print('STUDY', folder, flush=True)
    results = []
    for recipe in recipes:
        arm, width = recipe[:2]
        run = folder / arm
        run.mkdir()
        candidate = expand_hidden(original, width) if width > 24 else copy.deepcopy(original)
        migration = float(np.max(np.abs(candidate.predict(train.x) - original.predict(train.x))))
        if migration > 1e-9:
            raise ValueError('Expansion changed the parent prediction at initialization')
        start, cpu = perf_counter(), process_time()
        tail_penalty = TAIL_PENALTIES.get(arm, 0)
        traces = fit_study(candidate, train.x, train.y, groups, recipe, tail_penalty=tail_penalty)
        cpu = process_time() - cpu
        wall = perf_counter() - start
        expected_steps = recipe[2] * math.ceil(len(train.y) / original.meta['batch_size'])
        if candidate.meta['steps'] != original.meta['steps'] + expected_steps:
            raise ValueError('Unexpected continuation step count')
        destination = run / 'checkpoint.npz'
        candidate.save(destination)
        score = bearing_metrics(Model.load(destination).predict(val.x), val.y, val_groups)
        gate = bearing_gate(before, score)
        reference_tail = before['rul_ranges_minutes']['0-100']
        tail = score['rul_ranges_minutes']['0-100']
        tail_gate = (tail['rmse'] <= reference_tail['rmse'] and (tail['overestimate_p95'] or 0) <=
                     (reference_tail['overestimate_p95'] or 0))
        result = dict(zip(FIELDS, recipe)) | {
            'completed_epochs': len(traces), 'near_failure_overestimate_penalty': tail_penalty,
            'training_cpu_seconds': cpu, 'training_wall_seconds': wall,
            'elapsed_steps': expected_steps, 'migration_delta_minutes': migration,
            'checkpoint_sha256': digest(destination), 'checkpoint': str(destination),
            'after': score, 'gate_vs_B2': gate, 'gate_vs_B0': bearing_gate(untrained, score),
            'near_failure_gate_vs_B2': tail_gate,
        }
        write_json(run / 'result.json', result)
        with (run / 'history.jsonl').open('w', encoding='utf8') as f:
            for trace in traces:
                metrics = {k: v for k, v in trace.items() if k != 'epoch'}
                f.write(json.dumps({'step': trace['epoch'], 'metrics': metrics}, allow_nan=False) + '\n')
            metrics = {'validation/rmse': score['overall']['rmse'], 'validation/mae': score['overall']['mae'],
                       'validation/bearing_mean_rmse': score['bearing_mean_rmse'],
                       'validation/bearing_worst_rmse': score['bearing_worst_rmse'],
                       'training/cpu_seconds': cpu, 'decision/passed_B2_gate': float(gate['passes']),
                       'decision/passed_near_failure_gate': float(tail_gate),
                       'validation/near_failure_rmse': tail['rmse'],
                       'validation/near_failure_positive_p95': tail['overestimate_p95'] or 0}
            for device, metric in score['bearings'].items():
                metrics[f'bearing/{device}/rmse'] = metric['rmse']
            f.write(json.dumps({'step': recipe[2] + 1, 'metrics': metrics}, allow_nan=False) + '\n')
        results.append(result)
        write_json(folder / 'summary.json', {'registration': registration, 'runs': results})
        print(f"{arm}: RMSE {score['overall']['rmse']:.3f}, MAE {score['overall']['mae']:.3f}, "
              f"device {score['bearing_mean_rmse']:.3f}, gate {gate['passes']}, "
              f"near-failure gate {tail_gate}", flush=True)


if __name__ == '__main__':
    main()

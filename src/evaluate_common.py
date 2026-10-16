"""Common-position test scores and comparable model-only/end-to-end timing."""
import argparse
import json
import math
import pickle
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from models import build_model
from ngram import InterpolatedTrigram


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()


def hardware_metadata(device):
    meta = {'system': platform.platform(), 'architecture': platform.machine(),
            'processor': platform.processor(), 'torch_version': torch.__version__, 'device': str(device)}
    if platform.system() == 'Darwin':
        for key, command in [('model', ['sysctl', '-n', 'hw.model']),
                             ('chip', ['sysctl', '-n', 'machdep.cpu.brand_string']),
                             ('memory_bytes', ['sysctl', '-n', 'hw.memsize'])]:
            try:
                value = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
                meta[key] = int(value) if key == 'memory_bytes' else value
            except (OSError, subprocess.CalledProcessError, ValueError):
                pass
    return meta


def make_window(ids, position, context, mode):
    if mode == 'chunk_matched':
        start = ((position - 1) // context) * context
    else:
        start = max(0, position - context)
    return ids[max(start, position - context):position]


def batches(ids, context, batch_size, mode):
    """Yield right-padded common targets; chunk_matched mirrors train.py chunk resets."""
    for first in range(2, len(ids), batch_size):
        positions = range(first, min(len(ids), first + batch_size))
        rows, lengths, targets = [], [], []
        for i in positions:
            window = make_window(ids, i, context, mode)
            rows.append(window + [0] * (context - len(window)))
            lengths.append(len(window))
            targets.append(ids[i])
        yield rows, lengths, targets


def evaluate_neural(model, ids, context, device, batch_size=128, timing_repeats=3, mode='rolling', benchmark=True):
    model.eval()
    if not benchmark:
        loss_sum, target_count = 0.0, 0
        with torch.inference_mode():
            for rows, lengths, targets in batches(ids, context, batch_size, mode):
                x = torch.tensor(rows, dtype=torch.long, device=device)
                y = torch.tensor(targets, dtype=torch.long, device=device)
                logits = model.next_logits(x, lengths)
                loss_sum += F.cross_entropy(logits, y, reduction='sum').item()
                target_count += len(targets)
        return {'test_loss': loss_sum / target_count, 'test_ppl': math.exp(loss_sum / target_count),
                'test_tokens': target_count, 'timing': {'measured': False}}

    all_rows, all_lengths, all_targets = [], [], []
    for rows, lengths, targets in batches(ids, context, batch_size, mode):
        all_rows.extend(rows); all_lengths.extend(lengths); all_targets.extend(targets)
    x_cpu = torch.tensor(all_rows, dtype=torch.long)
    y_cpu = torch.tensor(all_targets, dtype=torch.long)

    # Warm the backend before timing; the same prebuilt workload measures forward-only cost.
    x_device = x_cpu.to(device)
    synchronize(device)
    with torch.inference_mode():
        model.next_logits(x_device[:min(batch_size, len(x_device))], all_lengths[:batch_size])
    synchronize(device)

    forward_times = []
    with torch.inference_mode():
        for _ in range(timing_repeats):
            synchronize(device); started = time.perf_counter()
            for first in range(0, len(x_device), batch_size):
                model.next_logits(x_device[first:first + batch_size], all_lengths[first:first + batch_size])
            synchronize(device); forward_times.append(time.perf_counter() - started)

    # End-to-end timing includes constructing windows, tensor creation/device transfer, forward pass, and loss.
    end_to_end_times, loss_sum, target_count = [], None, 0
    for repeat in range(timing_repeats):
        synchronize(device); started = time.perf_counter()
        current_loss, current_count = 0.0, 0
        with torch.inference_mode():
            for rows, lengths, targets in batches(ids, context, batch_size, mode):
                x = torch.tensor(rows, dtype=torch.long, device=device)
                y = torch.tensor(targets, dtype=torch.long, device=device)
                logits = model.next_logits(x, lengths)
                current_loss += F.cross_entropy(logits, y, reduction='sum').item()
                current_count += len(targets)
        synchronize(device); end_to_end_times.append(time.perf_counter() - started)
        if repeat == 0:
            loss_sum, target_count = current_loss, current_count

    forward_seconds = statistics.median(forward_times)
    end_to_end_seconds = statistics.median(end_to_end_times)
    return {'test_loss': loss_sum / target_count, 'test_ppl': math.exp(loss_sum / target_count),
            'test_tokens': target_count,
            'timing': {'forward_seconds_median': forward_seconds,
                       'forward_tokens_per_second': target_count / forward_seconds,
                       'end_to_end_seconds_median': end_to_end_seconds,
                       'end_to_end_tokens_per_second': target_count / end_to_end_seconds,
                       'repeats': timing_repeats, 'warmup_batches': 1,
                       'forward_scope': 'prebuilt device tensors; model forward only',
                       'end_to_end_scope': 'window construction, tensor creation/device transfer, forward, and cross-entropy'},
            # Compatibility aliases for existing summary notebooks.
            'inference_seconds': end_to_end_seconds,
            'tokens_per_second': target_count / end_to_end_seconds}


def evaluate_trigram(trigram, ids, timing_repeats=3):
    triples = [(ids[i - 2], ids[i - 1], ids[i]) for i in range(2, len(ids))]
    # Warm-up uses a small prefix, then both timing scopes repeat the exact same targets.
    for a, b, c in triples[:min(256, len(triples))]:
        trigram.prob(a, b, c)
    forward_times, end_to_end_times = [], []
    loss_sum = 0.0
    for repeat in range(timing_repeats):
        started = time.perf_counter()
        values = [-math.log(trigram.prob(a, b, c)) for a, b, c in triples]
        forward_times.append(time.perf_counter() - started)
        if repeat == 0:
            loss_sum = sum(values)
        started = time.perf_counter()
        total = 0.0
        for i in range(2, len(ids)):
            total -= math.log(trigram.prob(ids[i - 2], ids[i - 1], ids[i]))
        end_to_end_times.append(time.perf_counter() - started)
    n = len(triples)
    forward_seconds = statistics.median(forward_times)
    end_to_end_seconds = statistics.median(end_to_end_times)
    return {'test_loss': loss_sum / n, 'test_ppl': math.exp(loss_sum / n), 'test_tokens': n,
            'timing': {'forward_seconds_median': forward_seconds,
                       'forward_tokens_per_second': n / forward_seconds,
                       'end_to_end_seconds_median': end_to_end_seconds,
                       'end_to_end_tokens_per_second': n / end_to_end_seconds,
                       'repeats': timing_repeats, 'warmup_predictions': min(256, n),
                       'forward_scope': 'prebuilt (context, target) triples; probability computation only',
                       'end_to_end_scope': 'test-stream indexing, probability computation, and loss accumulation'},
            'inference_seconds': end_to_end_seconds, 'tokens_per_second': n / end_to_end_seconds}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed_3m_max_tokens_30k_vocab')
    parser.add_argument('--runs', default='runs/revised')
    parser.add_argument('--out', default='results/common_evaluation.json')
    parser.add_argument('--context', type=int, default=32)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--timing-repeats', type=int, default=3)
    parser.add_argument('--models', nargs='+', choices=['rnn', 'lstm', 'transformer'], default=['rnn', 'lstm', 'transformer'])
    args = parser.parse_args()
    if args.timing_repeats < 1:
        parser.error('--timing-repeats must be at least 1')
    device = torch.device(args.device if args.device != 'auto' else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    root, run_root = Path(args.data), Path(args.runs)
    vocab = json.loads((root / 'vocab.json').read_text())
    test = json.loads((root / 'test.json').read_text())
    train = json.loads((root / 'train.json').read_text())
    output = {'evaluation': 'common targets at positions >= 2; rolling context is up to 32 preceding tokens',
              'device': str(device), 'hardware': hardware_metadata(device), 'context': args.context,
              'inference_batch_size': 128, 'timing_repeats': args.timing_repeats,
              'context_comparison': 'rolling full window versus chunk_matched train.py reset boundaries',
              'models': {}}
    for kind in args.models:
        folder = run_root / kind
        config = json.loads((folder / 'config.json').read_text())
        # Fill in physical host details from the training config; evaluation and training
        # run on the same machine even when their compute backends differ.
        for key in ('model', 'chip', 'memory_bytes'):
            if key not in output['hardware'] and key in config.get('hardware', {}):
                output['hardware'][key] = config['hardware'][key]
        model = build_model(kind, len(vocab), config['emb'], config['hidden'],
                            config['heads'], config['layers'], max(args.context, 256)).to(device)
        model.load_state_dict(torch.load(folder / 'best.pt', map_location=device, weights_only=True))
        rolling = evaluate_neural(model, test, args.context, device, mode='rolling', timing_repeats=args.timing_repeats)
        chunk_matched = evaluate_neural(model, test, args.context, device, mode='chunk_matched', benchmark=False)
        context2 = evaluate_neural(model, test, 2, device, mode='rolling', benchmark=False)
        params = sum(p.numel() for p in model.parameters())
        saved = json.loads((folder / 'results.json').read_text()) if (folder / 'results.json').exists() else {}
        history_path = folder / 'history.csv'
        training_seconds = saved.get('training_seconds')
        training_basis = 'legacy saved run timing; inspect report for scope caveat'
        if history_path.exists():
            import csv
            with history_path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            if rows and 'train_seconds' in rows[0]:
                training_seconds = sum(float(row['train_seconds']) for row in rows)
                training_basis = 'sum of synchronized train-epoch times; excludes validation and startup'
        rolling['context_comparisons'] = {'rolling_32': rolling.copy(),
                                         'chunk_matched_32': chunk_matched,
                                         'rolling_2': context2}
        rolling['parameters'] = params
        rolling['serialized_size_bytes'] = (folder / 'best.pt').stat().st_size
        rolling['serialized_artifact'] = str(folder / 'best.pt')
        rolling['training_hardware'] = config.get('hardware', {'resolved_device': config.get('resolved_device', config.get('device'))})
        rolling['training_seconds'] = training_seconds
        rolling['training_time_basis'] = training_basis
        output['models'][kind] = rolling

    trigram = InterpolatedTrigram(len(vocab))
    started = time.perf_counter(); trigram.fit(train); fit_seconds = time.perf_counter() - started
    trigram_result = evaluate_trigram(trigram, test, args.timing_repeats)
    run_root.mkdir(parents=True, exist_ok=True)
    trigram_bytes = pickle.dumps(trigram, protocol=pickle.HIGHEST_PROTOCOL)
    (run_root / 'trigram.pkl').write_bytes(trigram_bytes)
    trigram_result.update({'training_seconds': fit_seconds,
                           'training_time_basis': 'single fit on the training stream',
                           'stored_unigrams': len(trigram.uni), 'stored_bigrams': len(trigram.bi),
                           'stored_trigrams': len(trigram.tri),
                           'serialized_size_bytes': len(trigram_bytes),
                           'serialized_artifact': str(run_root / 'trigram.pkl')})
    output['models']['trigram'] = trigram_result
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()

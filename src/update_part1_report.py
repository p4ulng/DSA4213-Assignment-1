"""Refresh the Part I Markdown/PDF report from the latest revised run artifacts."""

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ('trigram', 'rnn', 'lstm', 'transformer')
MODEL_NAMES = {'trigram': 'Interpolated trigram', 'rnn': 'RNN', 'lstm': 'LSTM',
               'transformer': 'Causal Transformer'}


def load_json(path):
    return json.loads(path.read_text())


def replace_marked(text, start, end, body):
    first = text.index(start) + len(start)
    last = text.index(end, first)
    return text[:first] + '\n\n' + body.strip() + '\n\n' + text[last:]


def loss_summary(run_root):
    lines = []
    observations = []
    for kind in ('rnn', 'lstm', 'transformer'):
        path = run_root / kind / 'history.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        best = min(rows, key=lambda row: float(row['val_loss']))
        final = rows[-1]
        lines.append(
            f"- **{MODEL_NAMES[kind]}:** lowest validation loss {float(best['val_loss']):.3f} "
            f"at epoch {best['epoch']}; final train/validation loss "
            f"{float(final['train_loss']):.3f}/{float(final['val_loss']):.3f}."
        )
        if int(best['epoch']) < int(final['epoch']) and float(final['val_loss']) > float(best['val_loss']):
            observations.append(f"{MODEL_NAMES[kind]} validation loss turns upward after epoch {best['epoch']}")
        elif int(best['epoch']) == int(final['epoch']):
            observations.append(f"{MODEL_NAMES[kind]} validation loss still improves at the end of the run")
        else:
            observations.append(f"{MODEL_NAMES[kind]} has a small validation-loss reversal after epoch {best['epoch']}")
    lines.append('')
    lines.append(' '.join(observations) + '.')
    lines.append('')
    lines.append('![Training and validation loss curves](../results/revised_loss_curves.png)')
    return '\n'.join(lines)


def build_results(data_root, run_root, common_path, generation_path):
    common = load_json(common_path)
    generation = load_json(generation_path)
    metadata = load_json(data_root / 'metadata.json')
    split_ids = {split: load_json(data_root / f'{split}.json') for split in ('train', 'validation', 'test')}
    unk_rates = {split: sum(token == 1 for token in ids) / max(1, len(ids))
                 for split, ids in split_ids.items()}

    lines = [
        f"The BPE vocabulary contains **{metadata['vocab_size']:,} pieces** (requested {metadata['requested_vocab_size']:,}). "
        f"The processed training split contains **{len(split_ids['train']):,} subword IDs** "
        f"and {max(0, len(split_ids['train']) - 1):,} next-token targets per full epoch; "
        f"the complete encoded training source had {metadata['train_source_tokens_before_cap']:,} IDs before the cap. "
        f"`<unk>` rates are {unk_rates['train']:.3%} train, {unk_rates['validation']:.3%} validation, "
        f"and {unk_rates['test']:.3%} test."
    ]
    lines.append('')
    lines.append('**Common test quality, model size, and inference cost:**')
    for kind in MODEL_ORDER:
        result = common['models'][kind]
        size_mb = result['serialized_size_bytes'] / 1_000_000
        timing = result['timing']
        if kind == 'trigram':
            size = f"{result['stored_unigrams'] + result['stored_bigrams'] + result['stored_trigrams']:,} stored n-gram entries"
        else:
            size = f"{result['parameters']:,} parameters"
        lines.append(
            f"- **{MODEL_NAMES[kind]}:** test CE {result['test_loss']:.3f} nats, "
            f"PPL {result['test_ppl']:.2f}; {size}; serialized size {size_mb:.2f} MB; "
            f"training {result.get('training_seconds', float('nan')):.2f} s; "
            f"inference {timing['forward_tokens_per_second']:,.0f} model-only and "
            f"{timing['end_to_end_tokens_per_second']:,.0f} end-to-end tokens/s."
        )
    lines.append('')
    lines.append(
        f"Scores use {common['models']['trigram']['test_tokens']:,} common test targets. "
        f"Training used {common['hardware'].get('chip', 'the recorded device')} with "
        f"{common['hardware'].get('memory_bytes', 0) / (1024**3):.0f} GB memory when hardware metadata is available; "
        f"neural training backend is recorded per model, and inference ran on {common['device'].upper()}. "
        "Inference figures are medians of three warmed passes; model-only timing uses prebuilt tensors and "
        "end-to-end timing includes window construction, tensor creation/transfer, forward computation, and loss."
    )
    lines.append('')
    lines.append('**Neural context sensitivity (perplexity):**')
    for kind in ('rnn', 'lstm', 'transformer'):
        comp = common['models'][kind]['context_comparisons']
        lines.append(
            f"- **{MODEL_NAMES[kind]}:** rolling 32 {comp['rolling_32']['test_ppl']:.2f}; "
            f"chunk-matched {comp['chunk_matched_32']['test_ppl']:.2f}; "
            f"context 2 {comp['rolling_2']['test_ppl']:.2f}."
        )
    lines.append('')
    lines.append(loss_summary(run_root))
    return '\n'.join(lines), generation


def appendix(generation):
    lines = [
        f"All samples use {generation['settings']['new_tokens']} new subword tokens maximum, "
        f"temperature {generation['settings']['temperature']}, top-k {generation['settings']['top_k']}, "
        f"and seed {generation['settings']['seed']}. Generation stops at `<eos>`; the output below is decoded text."
    ]
    prompts = [row['prompt'] for row in generation['continuations']['trigram']]
    for prompt_index, prompt in enumerate(prompts):
        lines.append('')
        lines.append(f'**Prompt:** “{prompt}”')
        for kind in MODEL_ORDER:
            sample = generation['continuations'][kind][prompt_index]
            continuation = sample['continuation'].strip() or '[stopped immediately at `<eos>`]'
            suffix = f" (stopped at `<eos>` after {sample['generated_tokens']} pieces)" if sample['stopped_at_eos'] else ''
            lines.append(f"- **{MODEL_NAMES[kind]}:** `{continuation}`{suffix}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 'processed_3m_max_tokens_30k_vocab')
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs' / 'revised')
    parser.add_argument('--common', type=Path, default=ROOT / 'results' / 'revised_common_evaluation.json')
    parser.add_argument('--generation', type=Path, default=ROOT / 'results' / 'revised_generation.json')
    parser.add_argument('--markdown', type=Path, default=ROOT / 'reports' / 'Part_I_Analysis.md')
    parser.add_argument('--pdf', type=Path, default=ROOT / 'reports' / 'Part_I_Analysis.pdf')
    args = parser.parse_args()

    results, generation = build_results(args.data, args.runs, args.common, args.generation)
    text = args.markdown.read_text()
    text = replace_marked(text, '<!-- AUTO_RESULTS_START -->', '<!-- AUTO_RESULTS_END -->', results)
    text = replace_marked(text, '<!-- AUTO_APPENDIX_START -->', '<!-- AUTO_APPENDIX_END -->', appendix(generation))
    args.markdown.write_text(text)
    subprocess.run([
        'pandoc', str(args.markdown), '--resource-path=reports:results:.', '--pdf-engine=xelatex',
        '-V', 'geometry:margin=0.65in', '-V', 'fontsize=9pt', '-o', str(args.pdf)
    ], cwd=ROOT, check=True)
    print(f'Updated {args.markdown} and {args.pdf}')


if __name__ == '__main__':
    main()

"""Prepare WikiText splits with a train-only SentencePiece BPE vocabulary."""

import argparse
import json
import random
from pathlib import Path

import sentencepiece as spm

VOCAB_SIZE = 30_000
MAX_TRAIN_TOKENS = 3_000_000
SEED = 42


def encode_text(processor, text, add_eos=False):
    """Encode text as subword IDs, optionally ending with the project's EOS ID."""
    ids = processor.encode(text, out_type=int)
    if add_eos and ids:
        ids.append(processor.eos_id())
    return ids


def prepare(splits, out, vocab_size=VOCAB_SIZE, max_train_tokens=MAX_TRAIN_TOKENS, seed=SEED):
    """Train a BPE tokenizer on train text and encode all splits with line EOS markers."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = {name: [line for line in values if line.strip()] for name, values in splits.items()}

    model_prefix = out / 'tokenizer'
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(lines['train']),
        model_prefix=str(model_prefix),
        model_type='bpe',
        vocab_size=vocab_size,
        character_coverage=1.0,
        normalization_rule_name='nmt_nfkc_cf',
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=2,
        pad_piece='<pad>',
        unk_piece='<unk>',
        eos_piece='<eos>',
        byte_fallback=True,
        hard_vocab_limit=True,
        shuffle_input_sentence=False,
        max_sentence_length=16384,
        num_threads=1,
    )
    processor = spm.SentencePieceProcessor(model_file=str(model_prefix) + '.model')
    if (processor.pad_id(), processor.unk_id(), processor.eos_id()) != (0, 1, 2):
        raise RuntimeError('SentencePiece special-token IDs do not match the model code.')

    encoded_lines = {
        name: [encode_text(processor, line, add_eos=True) for line in split_lines]
        for name, split_lines in lines.items()
    }
    source_train_tokens = sum(map(len, encoded_lines['train']))
    selected_indices = list(range(len(encoded_lines['train'])))
    selected_lines = len(selected_indices)
    if max_train_tokens and source_train_tokens > max_train_tokens:
        order = list(selected_indices)
        random.Random(seed).shuffle(order)
        selected_indices = []
        selected_count = 0
        for index in order:
            selected_indices.append(index)
            selected_count += len(encoded_lines['train'][index])
            if selected_count >= max_train_tokens:
                break
        selected_lines = len(selected_indices)
        selected_indices.sort()

    train_ids = []
    for index in selected_indices:
        remaining = max_train_tokens - len(train_ids) if max_train_tokens else None
        line_ids = encoded_lines['train'][index]
        if remaining is None or len(line_ids) <= remaining:
            train_ids.extend(line_ids)
        elif remaining > 0:
            train_ids.extend(line_ids[:remaining - 1])
            train_ids.append(processor.eos_id())
        if max_train_tokens and len(train_ids) >= max_train_tokens:
            break

    tokenized = {'train': train_ids}
    for name in ('validation', 'test'):
        tokenized[name] = [token for line_ids in encoded_lines[name] for token in line_ids]

    vocab = [processor.id_to_piece(i) for i in range(processor.get_piece_size())]
    for split, ids in tokenized.items():
        (out / f'{split}.json').write_text(json.dumps(ids))
        unk_count = sum(token_id == processor.unk_id() for token_id in ids)
        print(f'{split}: {len(ids):,} subword tokens; UNK {unk_count / max(1, len(ids)):.3%}')
    (out / 'vocab.json').write_text(json.dumps(vocab, indent=2))
    (out / 'metadata.json').write_text(json.dumps({
        'tokenizer': 'SentencePiece BPE; NFKC case-fold normalization; byte fallback',
        'tokenizer_model': str(model_prefix) + '.model',
        'vocab_size': len(vocab),
        'requested_vocab_size': vocab_size,
        'max_train_tokens': max_train_tokens,
        'train_source_tokens_before_cap': source_train_tokens,
        'train_nonempty_source_lines': len(encoded_lines['train']),
        'train_source_lines_selected': selected_lines,
        'train_sampling': 'seeded line sample across the source training split only when encoded train stream exceeds the configured token cap; selected lines restored to corpus order',
        'seed': seed,
        'special_token_ids': {'pad': processor.pad_id(), 'unk': processor.unk_id(), 'eos': processor.eos_id()},
        'eos_policy': 'append one EOS ID after each nonempty source line',
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Prepare WikiText-2 with SentencePiece BPE subwords.')
    parser.add_argument('--out', default='data/processed_3m_max_tokens_30k_vocab')
    parser.add_argument('--vocab-size', type=int, default=VOCAB_SIZE)
    parser.add_argument('--max-train-tokens', type=int, default=MAX_TRAIN_TOKENS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    if args.demo:
        lines = ['The scientist studied the language model.', 'The model learned from text.'] * 200
        splits = {'train': lines, 'validation': lines[:40], 'test': lines[40:80]}
    else:
        from datasets import load_dataset
        dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')
        splits = {name: dataset[name]['text'] for name in ('train', 'validation', 'test')}
    prepare(splits, args.out, args.vocab_size, args.max_train_tokens, args.seed)


if __name__ == '__main__':
    main()

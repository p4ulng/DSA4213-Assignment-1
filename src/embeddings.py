"""Create vocabulary-aligned embedding tables for Part II."""
import argparse
import json
import time
from pathlib import Path

import numpy as np


SPECIAL_TOKENS = {"<pad>", "<unk>", "<eos>"}


def split_at_eos(tokens, eos_token):
    """Keep each source line as a separate Skip-gram sentence, including its EOS."""
    sentences = []
    current = []
    for token in tokens:
        current.append(token)
        if token == eos_token:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed_3m_max_tokens_30k_vocab')
    parser.add_argument('--out', default='runs/embeddings')
    parser.add_argument('--method', choices=['skipgram', 'glove'], required=True)
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data)
    vocab = json.loads((data_dir / 'vocab.json').read_text())
    train_ids = json.loads((data_dir / 'train.json').read_text())
    words = [vocab[token_id] for token_id in train_ids]
    special_ids = {piece: vocab.index(piece) for piece in SPECIAL_TOKENS if piece in vocab}
    if '<pad>' not in special_ids:
        raise ValueError("Vocabulary must contain the '<pad>' token.")
    pad_id = special_ids['<pad>']
    eos_token = '<eos>'

    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    # Unmatched vocabulary entries get a small, reproducible fallback vector.
    matrix = rng.normal(0.0, 0.05, (len(vocab), args.dim)).astype(np.float32)
    matrix[pad_id] = 0.0
    found = set()

    if args.method == 'skipgram':
        from gensim.models import Word2Vec

        sentences = split_at_eos(words, eos_token)
        model = Word2Vec(
            sentences=sentences,
            vector_size=args.dim,
            window=args.window,
            min_count=1,
            workers=1,
            sg=1,
            negative=5,
            epochs=args.epochs,
            seed=args.seed,
        )
        for index, piece in enumerate(vocab):
            if piece in model.wv:
                matrix[index] = model.wv[piece]
                found.add(piece)
    else:
        import gensim.downloader as api

        model = api.load(f'glove-wiki-gigaword-{args.dim}')
        for index, piece in enumerate(vocab):
            # Special IDs are project-specific; never treat them as ordinary words.
            if piece in SPECIAL_TOKENS:
                continue
            # SentencePiece marks word-initial pieces with ▁; GloVe stores plain words.
            word = piece[1:] if piece.startswith('▁') else piece
            if word in model:
                matrix[index] = model[word]
                found.add(piece)

    # Keep padding semantically empty even if an embedding source contains a match.
    matrix[pad_id] = 0.0
    lexical_ids = [
        index for index, piece in enumerate(vocab)
        if piece not in SPECIAL_TOKENS and piece in found
    ]
    if lexical_ids:
        # Use the mean lexical vector as a deterministic neutral fallback for specials
        # that have no source-specific representation (notably GloVe's <eos>).
        fallback = matrix[lexical_ids].mean(axis=0)
        if '<unk>' in special_ids:
            matrix[special_ids['<unk>']] = fallback
        # Skip-gram learns EOS from the corpus; GloVe does not have project EOS tokens.
        if args.method == 'glove' and '<eos>' in special_ids:
            matrix[special_ids['<eos>']] = fallback

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / f'{args.method}_{args.dim}d.npy'
    np.save(matrix_path, matrix)

    coverage_vocab_size = sum(piece not in SPECIAL_TOKENS for piece in vocab)
    covered_types = len(lexical_ids)
    metadata = {
        'method': args.method,
        'dimension': args.dim,
        'window': args.window if args.method == 'skipgram' else None,
        'epochs': args.epochs if args.method == 'skipgram' else None,
        'vocab_size': len(vocab),
        'covered_types': covered_types,
        'coverage_vocab_size': coverage_vocab_size,
        'type_coverage': covered_types / max(coverage_vocab_size, 1),
        'matrix': str(matrix_path),
        'missing_initialization': 'seeded N(0, 0.05); <unk> uses mean matched lexical vector when available; GloVe <eos> uses that mean when available; <pad> zero',
        'special_token_policy': {
            '<pad>': 'excluded from source lookup and set to zero',
            '<unk>': 'excluded from source lookup and initialized to mean matched lexical vector when available',
            '<eos>': 'learned from the Skip-gram training stream; excluded from GloVe lookup and initialized to mean matched lexical vector when available',
        },
        'skipgram_sentence_policy': 'split at each <eos>; keep <eos> as the last token of its sentence',
        'seed': args.seed,
        'training_seconds': time.perf_counter() - started,
    }
    (out_dir / f'{args.method}_{args.dim}d.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()

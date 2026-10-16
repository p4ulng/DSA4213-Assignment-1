"""Frozen language-model representation comparison on AG News (Part III)."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from models import build_model
import sentencepiece as spm


def collate(batch, processor, max_len):
    sequences, labels, texts = [], [], []
    for item in batch:
        seq = processor.encode(item['text'], out_type=int)[:max_len]
        seq = seq or [processor.unk_id()]
        sequences.append(seq); labels.append(int(item['label'])); texts.append(item['text'])
    width = max(map(len, sequences))
    x = torch.zeros(len(batch), width, dtype=torch.long)
    lengths = []
    for i, seq in enumerate(sequences):
        x[i, :len(seq)] = torch.tensor(seq)
        lengths.append(len(seq))
    return x, torch.tensor(labels), lengths, texts


def features(model, dataset, processor, device, max_len, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate(b, processor, max_len))
    feats, labels, texts, unknown, token_count = [], [], [], 0, 0
    model.eval()
    with torch.inference_mode():
        for x, y, lengths, raw in loader:
            unknown += int((x == 1).sum()); token_count += sum(lengths)
            x = x.to(device)
            hidden = model.encode(x)
            mask = torch.arange(x.shape[1], device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            feats.append(pooled.cpu().numpy()); labels.extend(y.numpy().tolist()); texts.extend(raw)
    return np.concatenate(feats), np.asarray(labels), texts, unknown, token_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/processed_3m_max_tokens_30k_vocab'); p.add_argument('--runs', default='runs/revised')
    p.add_argument('--out', default='runs/downstream'); p.add_argument('--train-limit', type=int, default=20000)
    p.add_argument('--test-limit', type=int, default=0); p.add_argument('--max-len', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=128); p.add_argument('--device', default='auto')
    p.add_argument('--seed', type=int, default=42); a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    from datasets import load_dataset
    # Use the Hub's fully qualified repository ID; recent datasets/Hub versions
    # reject the old shorthand `ag_news` because it has no namespace.
    ds = load_dataset('fancyzhx/ag_news')
    train = ds['train'].shuffle(seed=a.seed)
    if a.train_limit: train = train.select(range(min(a.train_limit, len(train))))
    test = ds['test']
    if a.test_limit: test = test.select(range(min(a.test_limit, len(test))))
    root = Path(a.data); vocab = json.loads((root / 'vocab.json').read_text())
    processor = spm.SentencePieceProcessor(model_file=str(root / 'tokenizer.model'))
    device = torch.device(a.device if a.device != 'auto' else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    results = {'dataset': 'AG News', 'classes': ['World', 'Sports', 'Business', 'Sci/Tech'],
               'train_examples': len(train), 'test_examples': len(test), 'train_limit': a.train_limit,
               'max_tokens': a.max_len, 'representation': 'mean of valid contextual hidden states; frozen language models',
               'classifier': 'StandardScaler + multinomial LogisticRegression (C=1, max_iter=300)',
               'device': str(device), 'models': {}}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for kind in ('lstm', 'transformer'):
        folder = Path(a.runs) / kind
        config = json.loads((folder / 'config.json').read_text())
        model = build_model(kind, len(vocab), config['emb'], config['hidden'], config['heads'],
                            config['layers'], max(256, a.max_len)).to(device)
        model.load_state_dict(torch.load(folder / 'best.pt', map_location=device, weights_only=True))
        x_train, y_train, _, unk_train, n_train = features(model, train, processor, device, a.max_len, a.batch_size)
        x_test, y_test, texts, unk_test, n_test = features(model, test, processor, device, a.max_len, a.batch_size)
        classifier = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=300, random_state=a.seed))
        classifier.fit(x_train, y_train); pred = classifier.predict(x_test)
        result = {'accuracy': float(accuracy_score(y_test, pred)),
                  'macro_f1': float(f1_score(y_test, pred, average='macro')),
                  'classification_report': classification_report(y_test, pred, output_dict=True),
                  'train_unk_rate': unk_train / max(1, n_train), 'test_unk_rate': unk_test / max(1, n_test),
                  'errors': int((pred != y_test).sum())}
        errors_path = out / f'{kind}_errors.csv'
        with errors_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'gold', 'predicted']); writer.writeheader()
            for i in np.flatnonzero(pred != y_test)[:100]:
                writer.writerow({'text': texts[i], 'gold': int(y_test[i]), 'predicted': int(pred[i])})
        result['error_examples'] = str(errors_path); results['models'][kind] = result
        print(kind, result['accuracy'], result['macro_f1'])
    (out / 'results.json').write_text(json.dumps(results, indent=2)); print(json.dumps(results, indent=2))


if __name__ == '__main__': main()

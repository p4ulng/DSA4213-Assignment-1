"""Matched prompt/decoding comparison for the Part I language models."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import sentencepiece as spm

from models import build_model
from ngram import InterpolatedTrigram


def sample(probs, temperature, top_k):
    # Temperature adjusts randomness; top-k restricts sampling to the k highest probabilities.
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.power(np.maximum(probs, 1e-30), 1.0 / temperature)
    if top_k:
        keep = np.argpartition(probs, -top_k)[-top_k:]
        masked = np.zeros_like(probs); masked[keep] = probs[keep]; probs = masked
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))

def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()


def main():
    # Generate from identical prompts with shared sampling settings for a qualitative comparison.
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/processed_3m_max_tokens_30k_vocab'); p.add_argument('--runs', default='runs/revised')
    p.add_argument('--out', default='results/generation.json'); p.add_argument('--device', default='auto')
    p.add_argument('--new-tokens', type=int, default=30); p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=40); p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    root = Path(a.data); vocab = json.loads((root/'vocab.json').read_text())
    processor = spm.SentencePieceProcessor(model_file=str(root/'tokenizer.model'))
    device = torch.device(a.device if a.device != 'auto' else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    train = json.loads((root/'train.json').read_text())
    trigram = InterpolatedTrigram(len(vocab)); trigram.fit(train)
    # Fixed prompts ensure differences come from models rather than different starting text.
    prompts = ['The scientist studied', 'The government announced', 'In the history of']
    output = {'settings': {'new_tokens': a.new_tokens, 'temperature': a.temperature, 'top_k': a.top_k,
                          'seed': a.seed, 'device': str(device), 'tokenizer': 'SentencePiece BPE',
                          'eos_stopping': True, 'pad_suppressed': True}, 'continuations': {}}
    for kind in ('trigram', 'rnn', 'lstm', 'transformer'):
        rng_seed = a.seed
        generated = []
        # Neural checkpoints are loaded once per model; the trigram is already fit above.
        if kind != 'trigram':
            folder = Path(a.runs) / kind
            config = json.loads((folder/'config.json').read_text())
            model = build_model(kind, len(vocab), config['emb'], config['hidden'], config['heads'],
                                config['layers'], 256).to(device)
            model.load_state_dict(torch.load(folder/'best.pt', map_location=device, weights_only=True)); model.eval()
        for prompt_idx, prompt in enumerate(prompts):
            # SentencePiece decomposes rare prompt words into known subword units.
            history = processor.encode(prompt, out_type=int) or [processor.eos_id()]
            np.random.seed(rng_seed + prompt_idx)
            synchronize(device); t0 = time.perf_counter()
            generated_ids = []
            for _ in range(a.new_tokens):
                # Recompute next-token probabilities from current history, then sample one token.
                if kind == 'trigram':
                    # SentencePiece IDs are the same IDs used to train the trigram model.
                    padded = [processor.eos_id(), processor.eos_id()] + history
                    probs = [trigram.prob(*padded[-2:], c) for c in range(len(vocab))]
                else:
                    x = torch.tensor([history[-32:]], dtype=torch.long, device=device)
                    with torch.inference_mode():
                        logits = model.next_logits(x)[0].float().cpu().numpy()
                    logits -= logits.max(); probs = np.exp(logits)
                probs[processor.pad_id()] = 0.0
                next_id = sample(probs, a.temperature, a.top_k)
                if next_id == processor.eos_id():
                    break
                history.append(next_id)
                generated_ids.append(next_id)
            synchronize(device)
            generated.append({'prompt': prompt,
                              'continuation': processor.decode(generated_ids),
                              'text': processor.decode(history),
                              'generated_tokens': len(generated_ids),
                              'stopped_at_eos': len(generated_ids) < a.new_tokens,
                              'generation_seconds': time.perf_counter()-t0})
        output['continuations'][kind] = generated
    path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__': main()

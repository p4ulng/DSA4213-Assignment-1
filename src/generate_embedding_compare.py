"""Generate matched Part II continuations from the three LSTM embedding settings."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

from models import build_model


PROMPTS = ["The scientist studied", "The government announced", "In the history of"]
CONDITIONS = {
    "random_trainable": ("Part I LSTM; random input embeddings learned jointly", "runs/revised/lstm"),
    "skipgram_frozen": ("LSTM; corpus-trained Skip-gram input embeddings frozen", "runs/part2_revised/skipgram/lstm"),
    "glove_frozen": ("LSTM; public GloVe input embeddings frozen", "runs/part2_revised/glove/lstm"),
}


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def sample_next(logits, temperature, top_k, rng, pad_id):
    logits = logits.astype(np.float64) / temperature
    if top_k:
        keep = np.argpartition(logits, -top_k)[-top_k:]
        filtered = np.full_like(logits, -np.inf)
        filtered[keep] = logits[keep]
        logits = filtered
    logits[pad_id] = -np.inf
    logits -= np.max(logits)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed_3m_max_tokens_30k_vocab")
    parser.add_argument("--runs", default=".", help="project root containing the saved run folders")
    parser.add_argument("--out", default="results/part2_generation.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--new-tokens", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.new_tokens < 1 or args.temperature <= 0 or args.top_k < 1:
        parser.error("new-tokens and top-k must be positive, and temperature must be positive")

    root = Path(args.data)
    vocab = json.loads((root / "vocab.json").read_text())
    processor = spm.SentencePieceProcessor(model_file=str(root / "tokenizer.model"))
    device = torch.device(args.device if args.device != "auto" else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    run_root = Path(args.runs)
    output = {
        "settings": {"prompts": PROMPTS, "new_tokens_max": args.new_tokens,
                     "temperature": args.temperature, "top_k": args.top_k,
                     "seed": args.seed, "device": str(device),
                     "tokenizer": "SentencePiece BPE", "eos_stopping": True,
                     "pad_suppressed": True, "common_workload": True},
        "conditions": {},
    }

    for condition, (description, relative_folder) in CONDITIONS.items():
        folder = run_root / relative_folder
        config = json.loads((folder / "config.json").read_text())
        model = build_model("lstm", len(vocab), config["emb"], config["hidden"],
                            config["heads"], config["layers"], 256).to(device)
        model.load_state_dict(torch.load(folder / "best.pt", map_location=device, weights_only=True))
        model.eval()
        samples = []
        for prompt_index, prompt in enumerate(PROMPTS):
            history = processor.encode(prompt, out_type=int) or [processor.eos_id()]
            # Reinitialize the sampling stream per prompt and condition so every
            # condition receives the same seed while retaining its own probabilities.
            rng = np.random.default_rng(args.seed + prompt_index)
            generated_ids = []
            stopped_at_eos = False
            synchronize(device)
            started = time.perf_counter()
            for _ in range(args.new_tokens):
                x = torch.tensor([history[-32:]], dtype=torch.long, device=device)
                with torch.inference_mode():
                    logits = model.next_logits(x)[0].float().cpu().numpy()
                next_id = sample_next(logits, args.temperature, args.top_k, rng, processor.pad_id())
                if next_id == processor.eos_id():
                    stopped_at_eos = True
                    break
                history.append(next_id)
                generated_ids.append(next_id)
            synchronize(device)
            samples.append({"prompt": prompt,
                            "continuation": processor.decode(generated_ids),
                            "text": processor.decode(history),
                            "generated_tokens": len(generated_ids),
                            "stopped_at_eos": stopped_at_eos,
                            "generation_seconds": time.perf_counter() - started})
        output["conditions"][condition] = {"description": description,
                                           "checkpoint": str(folder / "best.pt"),
                                           "samples": samples}

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

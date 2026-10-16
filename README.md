# DSA4213 Assignment 1

This is an individual project assignment. It covers language modeling, word embeddings, RNNs, LSTMs,
Transformers, and learned representations. I compared model families on wikitext-2-raw-v1
examined the effect of embedding choices, and evaluated whether the learned representations help
a downstream task. 
On this corpus and under a fixed training budget, 
I seek to answer 3 questions:
How do (1) the sequence-model family and (2) the source of the input embeddings affect next-token prediction, 
and (3) does the better language model also provide better frozen features for text classification?


Most of the findings are found on the juypter notebooks. The notebooks run functions from the python files under src folder.


## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data and preprocessing

```bash
python src/data.py --out data/processed_3m_max_tokens_30k_vocab --max-train-tokens 3000000 --vocab-size 30000
```

SentencePiece BPE replaces the previous word-level tokenizer. It learns a 30,000-piece vocabulary from WikiText-2 training text, applies NFKC case-fold normalization and byte fallback, and appends `<eos>` after each nonempty source line. The train stream uses up to 3,000,000 subword IDs; if the encoded training corpus exceeds that cap, source lines are sampled reproducibly across the split and returned to source order before truncation. Validation and test remain complete. The saved `tokenizer.model` is shared by all models and used to decode generated text.

## Part1 - Train and compare models

```bash
python src/ngram.py --data data/processed_3m_max_tokens_30k_vocab
python src/train.py --model rnn --data data/processed_3m_max_tokens_30k_vocab --out runs/revised --epochs 25 --patience 5 --context 32 --batch-size 32 --seed 42
python src/train.py --model lstm --data data/processed_3m_max_tokens_30k_vocab --out runs/revised --epochs 25 --patience 5 --context 32 --batch-size 32 --seed 42
python src/train.py --model transformer --data data/processed_3m_max_tokens_30k_vocab --out runs/revised --epochs 25 --patience 5 --context 32 --batch-size 32 --seed 42
python src/evaluate_common.py --data data/processed_3m_max_tokens_30k_vocab --runs runs/revised --out results/revised_common_evaluation.json
python src/generate_compare.py --data data/processed_3m_max_tokens_30k_vocab --runs runs/revised --out results/revised_generation.json --new-tokens 30 --temperature 0.8 --top-k 40 --seed 42
python src/plot_part1_curves.py --runs runs/revised --out results/revised_loss_curves.png
python src/update_part1_report.py
```

Training replaces the previous word-token checkpoints under `runs/revised/<model>` with models trained on the SentencePiece IDs. It writes a configuration (including hardware metadata), epoch history, best validation checkpoint, and test result. The recorded training time sums synchronized training epochs and excludes validation, startup, and checkpoint I/O. Use `--max-batches` only for pilots; pilot metrics are truncated and not final estimates.

The causal Transformer is implemented as a stack of `nn.TransformerEncoderLayer` blocks with a strict upper-triangular future-token mask. The layer supplies self-attention and feed-forward sublayers; PyTorch's `nn.TransformerDecoderLayer` also includes cross-attention to an encoder memory, which this decoder-only language model does not use.

`evaluate_common.py` scores the same 272,267 test targets (positions 2 onward) for all four model families. Rolling 32-token and chunk-matched perplexities use these exact same targets; rolling may see the preceding 32 tokens from anywhere in the test stream, while chunk-matched resets at the training chunk boundaries. It also reports a neural context-2 ablation. Inference timings use three repeats after warm-up and report both prebuilt model-forward throughput and end-to-end throughput (window creation, tensor preparation/transfer, forward pass, and loss). Trigram context lookup and probability computation are separately timed on the same test targets. Model artifacts also report serialized size in bytes. Results are saved to `results/revised_common_evaluation.json`.

`generate_compare.py` evaluates three common prompts with up to 30 new subword tokens, temperature 0.8, top-k 40, and seed 42. Generation stops at `<eos>` and decodes pieces back into text. It replaces `results/revised_generation.json`.

Part II generation reuses those prompts and decoding settings for the Part I random-trained LSTM, frozen Skip-gram LSTM, and frozen GloVe LSTM:

```bash
python src/generate_embedding_compare.py --data data/processed_3m_max_tokens_30k_vocab --runs . --out results/part2_generation.json --new-tokens 30 --temperature 0.8 --top-k 40 --seed 42
```

## Part II: embedding variants

The LSTM trainer accepts a vocabulary-aligned embedding matrix and can freeze it. Reuse `runs/revised/lstm/best.pt` for jointly learned random embeddings, then train the two fixed-vector variants:

```bash
python src/embeddings.py --method skipgram --data data/processed_3m_max_tokens_30k_vocab --out runs/embeddings --dim 100 --window 5 --epochs 5 --seed 42
python src/embeddings.py --method glove --data data/processed_3m_max_tokens_30k_vocab --out runs/embeddings --dim 100 --seed 42
python src/train.py --model lstm --data data/processed_3m_max_tokens_30k_vocab --out runs/part2/skipgram --epochs 25 --patience 5 --context 32 --batch-size 32 --seed 42 --embedding-init runs/embeddings/skipgram_100d.npy --freeze-embedding
python src/train.py --model lstm --data data/processed_3m_max_tokens_30k_vocab --out runs/part2/glove --epochs 25 --patience 5 --context 32 --batch-size 32 --seed 42 --embedding-init runs/embeddings/glove_100d.npy --freeze-embedding
python src/evaluate_common.py --data data/processed_3m_max_tokens_30k_vocab --runs runs/part2/skipgram --models lstm --out results/part2_skipgram.json
python src/evaluate_common.py --data data/processed_3m_max_tokens_30k_vocab --runs runs/part2/glove --models lstm --out results/part2_glove.json
```

The public vectors are GloVe Wiki-Gigaword 100d, downloaded by Gensim on first use. The vector table was pretrained on Wikipedia 2014 and Gigaword 5 (about 6B tokens); this is prior information carried by the supplied embeddings, not extra text used in the project-side LSTM fitting. Both variants report vocabulary coverage; missing entries use seeded random vectors, `<unk>` is the mean available vector, and `<pad>` is zero. Embedding training/loading and alignment cost is recorded separately from language-model training. Public GloVe's original pretraining cost is not included.

## Part III: downstream representations

The downstream script compares frozen Part I LSTM and Transformer representations on AG News topic classification. It mean-pools valid contextual states and trains the same standardized logistic-regression classifier for each model.

```bash
python src/downstream.py --data data/processed_3m_max_tokens_30k_vocab --runs runs/revised --out runs/downstream --train-limit 20000 --max-len 128 --seed 42
```

It reports accuracy, macro-F1, unknown-token rates, per-class metrics, and up to 100 classification errors per model. AG News downloads on first use.

## Reproducibility and limitations

The default neural setup uses a 32-token context, batch size 32, Adam at 0.001, seed 42, and the available MPS device (or CPU). Training batches restart context at chunk boundaries; the chunk-matched evaluation makes this mismatch measurable alongside rolling-window evaluation. Checkpoints are selected by validation loss. Cite WikiText-2, any adapted course materials, and substantial coding assistance in the report as required by the assignment. The machine-readable run metadata records OS, architecture, chip/model when available, memory, PyTorch version, and device backend.

The notebook workflow is `01_Data_Preprocessing.ipynb`, `02_Ngram_Language_Model.ipynb`, `03_Neural_Language_Models.ipynb`, and `04_Part_2_onwards.ipynb`. Run notebook 01 first to prepare `data/processed_3m_max_tokens_30k_vocab`, then notebook 03 for Part I training and evaluation. Notebook 04 compares the Part I random-embedding LSTM against frozen Skip-gram and GloVe variants and continues into Part III; its training cells are opt-in.

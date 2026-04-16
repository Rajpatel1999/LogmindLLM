# LogMind

LogMind is a compact **GPT-style language model** implemented **from scratch in PyTorch**, intended for **learning how LLMs work** and as a stepping stone toward **understanding backend logs** and supporting **debugging workflows**. It avoids high-level “magic” wrappers so the core mechanics (attention, blocks, training, sampling) stay explicit and inspectable.

---

## Motivation

- **Understand internals**: Build intuition for embeddings, attention, residuals, and how next-token prediction is trained end to end.  
- **Apply to real problems**: Move toward models that can read **logs, traces, and error patterns** and help engineers **narrow down root causes** faster than ad-hoc string search alone.

---

## Features

- **Character level tokenizer**: Maps raw text to token IDs and back.  
- **Custom dataset pipeline**: Sliding window sequences with input/target shifted by one token for language modeling.  
- **Self attention from scratch**: Scaled dot product attention with learned Q/K/V projections.  
- **Transformer block**: Pre-norm attention + residual, then feed-forward (expand → ReLU → project) + residual.  
- **GPT style model**: Token and positional embeddings, stacked blocks, final layer norm, vocabulary logits (no softmax in the forward pass).  
- **Training loop**: Adam, cross-entropy on flattened logits vs targets, backprop over multiple epochs.  
- **Text generation**: Inference with softmax + `torch.multinomial` sampling from the last position.

---

## Architecture Overview

End to end flow:

**Text → Tokenizer → Dataset → GPT (Transformer blocks) → Logits → Sampled tokens → Decoded text**

1. Raw text is **tokenized** into integer IDs.  
2. The **dataset** yields `(input, target)` windows for next-token learning.  
3. The **model** maps token sequences to **logits** over the vocabulary at each position.  
4. During **training**, logits are compared to targets with cross-entropy.  
5. During **generation**, the model is run autoregressively; the **last** position’s distribution is sampled and appended until a length budget is reached, then **decoded** back to text.

---

## Project Structure

| Path | Role |
|------|------|
| `data/` | `TextDataset`: sliding windows over encoded text. |
| `models/` | `attention.py` (self-attention), `transformer_block.py`, `gpt_model.py`. |
| `training/` | `train.py`: DataLoader, optimizer, loss, epoch loop. |
| `inference/` | `generate.py`: autoregressive sampling from a trained checkpoint in memory. |
| `utils/` | `CharTokenizer`: character vocabulary, encode/decode. |
| `main.py` | **Entry point**: builds tokenizer, dataset, model, trains, then runs generation. |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Alternatively, from the repo root:

```bash
sh run.sh
```

---

## Example Output

Training prints **loss per epoch**; generation prints a **single sampled continuation** (quality is limited by data size and model capacity). Example shape of output after `main.py`:

```text
Epoch 10/10  loss: 0.72
Generated: API rered AfPI I API od
```

Your exact string will vary with **random sampling** and training noise.

---

## Current Limitations

- **Small dataset**: Demo-sized corpus; not enough signal for fluent or task-specific log reasoning.  
- **Character-level tokenizer**: Long contexts, weak word boundaries, no subword efficiency.  
- **No large-scale training**: Tiny model and few epochs; no distributed or mixed-precision pipeline.

---

## Future Improvements

- Train on **larger corpora** (e.g. TinyStories-style data or **curated log snippets**).  
- **Sampling controls**: temperature, top-*k*, top-*p* for more stable or diverse generations.  
- **FastAPI** (or similar) **HTTP API** for interactive chat / log Q&A.  
- **Richer tokenizer**: BPE or WordPiece for better compression and generalization.

---

## Tech Stack

- **Python**  
- **PyTorch**

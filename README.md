# Federated Fine-Tuning of LLMs via LoRA Adapters

**Team 16 — PaperRex** | CECS 574 | California State University, Long Beach | Spring 2026

> Krish Patel · Ayush Alpesh Gupta

A simulation study of federated fine-tuning using Low-Rank Adaptation (LoRA) adapters, implemented with the [Flower](https://flower.dev) federated learning framework and applied to the **Microsoft Phi-2 (2.7B)** model on the **Alpaca-GPT4** instruction-following dataset.

---

## Overview

Fine-tuning large language models (LLMs) on private, distributed data is critical for domains such as healthcare, legal, and finance — where centralizing data is legally or operationally infeasible. This project demonstrates that federated LoRA fine-tuning with 4-bit quantization (QLoRA) can achieve a **114× reduction in per-round communication cost** (~94.4 MB vs. ~10.8 GB for full-weight transmission) while producing consistent training loss reduction across heterogeneous client distributions.

### Key Findings

| Metric | IID | Non-IID | High Client Frac. | More Rounds |
|---|---|---|---|---|
| Final Val. Loss | 0.990 | 1.020 | 0.993 | 0.999 |
| Token Accuracy | 1.83% | 1.91% | 2.21% | 2.49% |
| Fairness Gap | 0.053 | 0.292 | 0.213 | 0.259 |
| Communication | 540 MB | 540 MB | 540 MB | 900 MB |

- Non-IID data amplifies the client fairness gap by **5.5×** vs. IID
- Increasing participation from 50% → 75% reduces the gap by **27%**
- Adding rounds yields diminishing generalization returns beyond round 6

---

## Repository Structure

```
.
├── CECS574_flower_federated_lora_simulation.ipynb  # Main experiment notebook
├── Team_16_PaperRex_paper.pdf                      # Final paper
├── Team_16_PaperRex_Presentation.pptx              # Slides
├── plots/
│   ├── loss_vs_round.png
│   ├── perplexity_vs_round.png
│   ├── fairness_vs_round.png
│   ├── token_vs_accuracy.png
│   └── comm_vs_round.png
└── results/
    ├── experiment_results.csv     # Final metrics for all 4 runs
    ├── experiment_results.json
    └── round_metrics_last_run.csv
```

---

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU with ≥ 8 GB VRAM (16 GB+ recommended; experiments run on A100 40 GB)
- Google Colab Pro or equivalent

### Install dependencies

```bash
pip install \
  flwr==1.x \
  transformers \
  peft \
  bitsandbytes \
  datasets \
  accelerate \
  trl
```

Or open the notebook directly in Google Colab — all installs are handled in the first cell.

---

## Running the Experiments

Open `CECS574_flower_federated_lora_simulation.ipynb` in Google Colab Pro (A100 runtime recommended).

The notebook is organized into the following sections:

1. **Configuration** — set model, dataset, LoRA, and FL hyperparameters
2. **Dataset formatting** — Alpaca-GPT4 prompt template construction
3. **Data partitioning** — IID and non-IID (semantic clustering) splits across 8 clients
4. **Flower client/server** — `FlowerClient` with `fit()`, `evaluate()`, and `get_parameters()`
5. **Experiment loop** — runs all 4 configurations and saves metrics to `results/`
6. **Visualization** — generates all 5 plots saved to `plots/`

To reproduce a specific experiment, set the configuration at the top of the notebook:

```python
PARTITION     = "non_iid"   # "iid" or "non_iid"
CLIENTS_TOTAL = 8
CLIENTS_ROUND = 4           # 4 (50%) or 6 (75%)
NUM_ROUNDS    = 6           # 6 or 10
```

---

## Model & LoRA Configuration

| Parameter | Value |
|---|---|
| Base model | `microsoft/phi-2` (2.7B) |
| Quantization | 4-bit NF4 (bitsandbytes) |
| LoRA rank `r` | 8 |
| LoRA alpha `α` | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `dense` |
| Trainable params | 11,796,480 (0.42% of model) |
| Local epochs | 1 per round |
| Batch size | 2 |
| Learning rate | 2×10⁻⁴ (linear warmup) |
| Max seq. length | 256 tokens |
| Optimizer | 8-bit paged AdamW |

---

## Results

All numerical results are saved in `results/experiment_results.csv`. Plots are in `plots/`.

### Communication Efficiency

Each round transmits ~94.4 MB of LoRA adapter weights (vs. ~10.8 GB for full Phi-2 weights in FP32), achieving a **114× communication reduction**. Cumulative cost scales linearly at ~94.4 MB/round.

### Fairness Under Non-IID Data

The client validation loss gap (max − min across clients) starts at 0.787 in round 1 and falls to 0.116 by round 5 — an 85% reduction — demonstrating that FedAvg aggregation progressively redistributes knowledge to data-scarce clients.

---

## Citation

```bibtex
@article{patel2026fedlora,
  title   = {Federated Fine-Tuning of Large Language Models via LoRA Adapters:
             A Flower-Based Simulation Study},
  author  = {Patel, Krish and Gupta, Ayush Alpesh},
  year    = {2026},
  note    = {CECS 574, California State University, Long Beach}
}
```

---

## License

This project is released for academic and educational purposes only.

# Reasoning SLM - Post-Training Qwen2.5-0.5B on GSM8K

This project takes **Qwen2.5-0.5B** (base) through supervised fine-tuning (SFT) and then GRPO on **GSM8K**, and measures what each stage adds against Qwen's own **Qwen2.5-0.5B-Instruct** (evaluation only, never trained).

## Setup

| Piece | Choice |
| --- | --- |
| Model | [`Qwen2.5-0.5B`](https://huggingface.co/Qwen/Qwen2.5-0.5B) (base) |
| Reference | [`Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| Data | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), config `main` (7,473 train / 1,319 test) |
| Training | TRL `SFTTrainer`, `GRPOTrainer` + PEFT (LoRA) |
| Hardware | Google Colab, L4 GPU (A100 for Stage 3) |

**Answer format.** Reasoning steps, then a final line `#### <number>`, the format GSM8K already uses. The same format is used for the SFT targets, the GRPO reward and eval extraction.

## Stages

| Stage | Notebook | Status |
| --- | --- | --- |
| Stage 1: Eval harness and baselines | [`notebooks/01_baseline.ipynb`](notebooks/01_baseline.ipynb) | Done |
| Stage 2: LoRA SFT | [`notebooks/02_sft.ipynb`](notebooks/02_sft.ipynb) | Done |
| Stage 3: GRPO on top of SFT | [`notebooks/03_grpo.ipynb`](notebooks/03_grpo.ipynb) | Done |
| Final: Test set evaluation | [`notebooks/04_test_eval.ipynb`](notebooks/04_test_eval.ipynb) | Done |

Final results and findings: [`writeup.md`](writeup.md).

## Repository Structure

```
├── notebooks/          # One Colab notebook per stage
├── results/            # JSON outputs of every experiment, one folder per stage (plus test/)
├── src/                # Code shared between notebooks
│   ├── config.py       # Model ids, dataset, seed, paths
│   ├── data/           # GSM8K loading and preprocessing
│   ├── prompts/        # Few-shot and chat prompt builders
│   ├── evaluation/     # Answer extraction, generation, scoring, general check
│   └── training/       # GRPO rewards, adapter merging
├── requirements.txt
└── writeup.md          # Final write-up
```

## Running on Colab

Each notebook starts by cloning this repository and installing the requirements, so everything in `src/` can be imported:

```python
!git clone -q https://github.com/JuaniLlaberia/slm-post-training.git /content/slm-post-training
%cd /content/slm-post-training
!pip install -q -r requirements.txt

from src.evaluation.harness import evaluate
```

Stage 3 also installs vLLM inside its notebook (not in `requirements.txt`, since it replaces Colab's torch) and needs a runtime restart after that install.

Results are written to `results/<stage>/`. Download them after each run, because `/content` is wiped when the runtime disconnects.

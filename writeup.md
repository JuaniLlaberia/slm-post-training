# Reasoning SLM - Write-up

## Setup

- **Model:** `Qwen2.5-0.5B` (base), post-trained with LoRA SFT and then GRPO.
- **Reference:** `Qwen2.5-0.5B-Instruct`, evaluated only, never trained.
- **Data:** GSM8K (`openai/gsm8k`, config `main`). A seeded 200-problem subset held out from train (seed 42) is used for iterating; the full 1,319-problem test set is kept for final numbers.
- **Answer format:** reasoning steps, then a final line `#### <number>`. The same format is used for the SFT targets, the GRPO reward and eval extraction.
- **Extraction:** the number after the last `####` (falling back to the last `\boxed{...}`), commas removed, compared to the gold as a number. No answer found counts as wrong.
- **Eval settings:** greedy decoding, at most 512 new tokens, stop at `\nQuestion:`, the same for every model.
- **Prompts:** the base model gets 8 solved train examples as a few-shot prompt; the instruct model gets its chat template, zero-shot, with an instruction to end with `#### <number>`.
- **General check:** 20 fixed non-math prompts, judged by hand. **Content** judges only the first answer against its rule; **clean** also requires that the model stopped right after answering.

All generations are saved in [`results/`](results/), one folder per stage.

## Results

Accuracy on the 200-problem validation subset (±3.5 pts). General check as content / clean passes out of 20.

| Model | Prompt | Subset acc. (200) | Full test acc. (1,319) | General check (/20) | No-answer rate | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B base | 8-shot | 48% |  | - | 4% | 2% hit the token limit |
| Qwen2.5-0.5B base | zero-shot | 0% (48.5% lenient) |  | 16 / 6 | 96% | 54.5% hit the token limit |
| + SFT (LoRA) | zero-shot |  |  |  |  |  |
| + SFT + GRPO | zero-shot |  |  |  |  |  |
| Qwen2.5-0.5B-Instruct | chat, zero-shot | 53% |  | 17 / 16 | 23.5% | 6.5% hit the token limit |

## What Each Stage Added

### Stage 1: Baselines

All three setups solve roughly the same share of problems, about 48-53%, within noise. What changes is behavior:

- **Base, zero-shot** scores 0%: 96% of outputs contain no `####` line, and 54.5% run until the 512-token limit. The model answers in its own words ("The answer is 16") and then keeps generating. A lenient check that reads the first "answer is N" finds 97/200 (48.5%) correct but unformatted answers. It knows the answers but not the expected format or when to stop.
- **Base, 8-shot** reaches 48% with a 4% no-answer rate: 8 examples are enough to teach it the format.
- **Instruct** reaches 53%: Qwen's post-training teaches it to follow instructions, stop, and reason step by step.

The knowledge comes from pretraining. Post-training and prompting decide whether it shows up in a usable form.

**Prediction for Stage 2:** SFT should bring zero-shot accuracy to about 48% by fixing format and stopping. A clearly higher score would mean SFT adds more than format.

### Stage 2: SFT

### Stage 3: GRPO

## What Specializing Cost

Baseline on the general check, before any training:

- **Base (zero-shot):** 16/20 content, 6/20 clean. It knows most answers but rarely stops.
- **Instruct:** 17/20 content, 16/20 clean. It stops reliably, which is where it differs from the base. Content is about the same.

Not stopping is what SFT should fix (EOS at the end of each completion), while content is what math specialization may erode.

## Examples

## What Surprised Me

- The base model's appended text in the general check (an "AI assistant" system prompt, "single-select problem" blocks) is instruction-tuning data from pretraining showing through, which suggests the "base" model is not a clean base.
- The instruct model applies step-by-step reasoning even to trivia ("To determine..."), a habit its post-training carried into every kind of task.

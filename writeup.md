# Reasoning SLM - Write-up

## Setup

- **Model:** `Qwen2.5-0.5B` (base), post-trained with LoRA SFT and then GRPO.
- **Reference:** `Qwen2.5-0.5B-Instruct`, evaluated only, never trained.
- **Data:** GSM8K (`openai/gsm8k`, config `main`). A seeded 499-problem subset held out from train (seed 42) is used for iterating; the full 1,319-problem test set is kept for final numbers.
- **Answer format:** reasoning steps, then a final line `#### <number>`. The same format is used for the SFT targets, the GRPO reward and eval extraction.
- **Extraction:** the number after the last `####` (falling back to the last `\boxed{...}`), commas removed, compared to the gold as a number. No answer found counts as wrong.
- **Eval settings:** greedy decoding, at most 512 new tokens, stop at `\nQuestion:`, the same for every model.
- **Prompts:** the base model gets 8 solved train examples as a few-shot prompt; the instruct model gets its chat template, zero-shot, with an instruction to end with `#### <number>`.
- **General check:** 20 fixed non-math prompts, judged by hand. **Content** judges only the first answer against its rule; **clean** also requires that the model stopped right after answering.

All generations are saved in [`results/`](results/), one folder per stage.

## Results

Accuracy on the 499-problem validation subset (±2.2 pts). General check as content / clean passes out of 20.

| Model | Prompt | Subset acc. (499) | Full test acc. (1,319) | General check (/20) | No-answer rate | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B base | 8-shot | 46.9% |  | - | 4.6% | 2.4% hit the token limit |
| Qwen2.5-0.5B base | zero-shot | 0.2% (48.3% lenient) |  | 16 / 6 | 95.2% | 55.1% hit the token limit |
| + SFT (LoRA) | zero-shot | 46.5% |  | 8 / 3 | 1.2% | 1.2% hit the token limit |
| + SFT + GRPO | zero-shot |  |  |  |  |  |
| Qwen2.5-0.5B-Instruct | chat, zero-shot | 53.3% (61.5% lenient) |  | 17 / 16 | 22.4% | 5.0% hit the token limit |

## What Each Stage Added

### Stage 1: Baselines

The base model solves about the same share of problems with or without examples, about 47-48%. The instruct model is ahead:

- **Base, zero-shot** scores 0.2%: 95.2% of outputs contain no `####` line, and 55.1% run until the 512-token limit. The model answers in its own words ("The answer is 16") and then keeps generating. A lenient check that reads the first "answer is N" finds 241/499 (48.3%) correct but unformatted answers. It knows the answers but not the expected format or when to stop.
- **Base, 8-shot** reaches 46.9% with a 4.6% no-answer rate: 8 examples are enough to teach it the format.
- **Instruct** reaches 53.3%, and about 61.5% counting the 41/499 no-answer outputs that end with the correct number in plain text ("Final Answer: 3 months"). Qwen's post-training teaches it to follow instructions, stop, and reason step by step.

For the base model, the knowledge comes from pretraining, and prompting decides whether it shows up in a usable form. The instruct model's gap is well outside the ±2.2 noise, so it may not be format alone.

**Prediction for Stage 2:** SFT should bring zero-shot accuracy to about 48% by fixing format and stopping. A clearly higher score would mean SFT adds more than format.

### Stage 2: SFT

LoRA (rank 16, all linear layers) on the 6,974 train solutions, zero-shot prompt, loss on the solution only, EOS at the end of each completion. The best checkpoint by eval loss is step 400, near the end of epoch 1; a second epoch only memorized the training set.

- **SFT, zero-shot** reaches 46.5% with a 1.2% no-answer rate and 1.2% hitting the token limit (base zero-shot: 95.2% and 55.1%).
- That matches base 8-shot (46.9%) and the base model's lenient zero-shot score (48.3%), as predicted.

SFT changed format only. Base and SFT solve the same share of problems, about 47-48%.

Qwen's post-training also improved the math. About 61.5% against about 48% is roughly 13 points, far outside the ±2.2 noise. This corrects the Stage 1 conclusion: Qwen's instruct model isn't just a better-formatted base model. Its post-training (much more data, math-specific tuning and RL) taught it to solve more problems.

**Target for Stage 3:** can GRPO move the model from about 47% toward about 61%? Unlike SFT, RL rewards correct answers, not imitating someone else's text, so it's the stage that could add real math ability.

### Stage 3: GRPO

## What Specializing Cost

Baseline on the general check, before any training:

- **Base (zero-shot):** 16/20 content, 6/20 clean. It knows most answers but rarely stops.
- **Instruct:** 17/20 content, 16/20 clean. It stops reliably, which is where it differs from the base. Content is about the same.

Not stopping is what SFT should fix (EOS at the end of each completion), while content is what math specialization may erode.

After SFT:

- **SFT (zero-shot):** 8/20 content, 3/20 clean. Half the content passes are gone, mostly from the math format leaking: 7 answers contain a `####` line (`#### Tokyo`), which fails both marks. Stopping didn't improve outside math: it still appends "You are an AI assistant…", and on open-ended writing it repeats or lists until the token limit. It learned to stop after a `####` line, not after an answer.

## Examples

## What Surprised Me

- The base model's appended text in the general check (an "AI assistant" system prompt, "single-select problem" blocks) is instruction-tuning data from pretraining showing through, which suggests the "base" model is not a clean base.
- The instruct model applies step-by-step reasoning even to trivia ("To determine..."), a habit its post-training carried into every kind of task.

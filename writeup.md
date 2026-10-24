# Reasoning SLM - Write-up

## Setup

- **Model:** `Qwen2.5-0.5B` (base), post-trained with LoRA SFT and then GRPO.
- **Reference:** `Qwen2.5-0.5B-Instruct`, evaluated only, never trained.
- **Data:** GSM8K (`openai/gsm8k`, config `main`). A seeded 499-problem subset held out from train (seed 42) is used for iterating; the full 1,319-problem test set is kept for final numbers.
- **Answer format:** reasoning steps, then a final line `#### <number>`. The same format is used for the SFT targets, the GRPO reward and eval extraction.
- **Extraction:** the number after the last `####` (falling back to the last `\boxed{...}`), commas removed, compared to the gold as a number. No answer found counts as wrong.
- **Eval settings:** greedy decoding, at most 512 new tokens, stop at `\nQuestion:`, the same for every model.
- **Prompts:** the base model gets 8 solved train examples as a few-shot prompt; the SFT and GRPO models get the zero-shot `Question: … Answer:` prompt they were trained on; the instruct model gets its chat template, zero-shot, with an instruction to end with `#### <number>`.
- **General check:** 20 fixed non-math prompts, judged by hand. **Content** judges only the first answer against its rule; **clean** also requires that the model stopped right after answering.

All generations are saved in [`results/`](results/), one folder per stage.

## Results

Accuracy on the 499-problem validation subset (±2.2 pts) and the full 1,319-problem test set (±1.4 pts). General check as content / clean passes out of 20; no-answer rate on validation. Validation scores are inflated for every model (see [Test Set and Possible Contamination](#test-set-and-possible-contamination)).

| Model | Prompt | Subset acc. (499) | Full test acc. (1,319) | General check (/20) | No-answer rate | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B base | 8-shot | 46.9% | 36.0% | - | 4.6% | 2.4% hit the token limit |
| Qwen2.5-0.5B base | zero-shot | 0.2% (48.3% lenient) | - | 16 / 6 | 95.2% | 55.1% hit the token limit; not run on test |
| + SFT (LoRA) | zero-shot | 46.5% | 35.0% | 8 / 3 | 1.2% | 1.2% hit the token limit |
| + SFT + GRPO | zero-shot | 64.7% | 47.7% | 10 / 5 | 0.0% | Step 400; 0.0% hit the token limit |
| Qwen2.5-0.5B-Instruct | chat, zero-shot | 53.3% (61.5% lenient) | 38.8% (46.9% lenient) | 17 / 16 | 22.4% | 5.0% hit the token limit |

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

A fresh LoRA (rank 16, no dropout) on the merged SFT model, trained with GRPO on the train problems: 8 sampled answers per problem at temperature 0.8, reward 1/0 for a correct answer (the eval extractor, after the same stop-string cut) plus 0.1 for a final `#### <number>` line. Learning rate 1e-5, 800 steps of 8 problems (about 0.9 epoch), `beta=0.001` (only to log KL), `loss_type="dapo"`, `scale_rewards="group"`.

- **Smoke run first** (100 steps at 5e-6): stable, and the sampled reward rose from about 25% to 34%, but greedy validation accuracy didn't move (46.1% and 48.7% at steps 50 and 100). The rising reward was the model sharpening its sampling, not solving new problems.
- **Full run:** the sampled reward rises from 34% to about 60% and plateaus from about step 500, while KL keeps growing. Entropy falls from 0.36 to 0.14 without collapsing, and completions get about 23% longer.
- **Validation** climbs to 64.7% at step 400, then plateaus at 64.7-66.7% through step 800. Every checkpoint from step 200 on is clearly above SFT (paired `z` 4.2 to 7.9): GRPO solves about 130 problems SFT misses and loses about 32.
- **Checkpoint choice:** the earliest on the plateau, unless a later one is clearly better problem by problem. None is (`z` at most 1.36), so step 400, with the least drift from SFT (KL about 0.13).

GRPO added real math ability: 46.5% to 64.7% on validation, +18.2 points (paired `z` = 7.2), with no-answer at 0.0%. That is clearly above Qwen's instruct model as scored (53.3%) and close to it once its format is fixed (61.5%). SFT taught the format by imitating reference solutions; GRPO, rewarded only for correct answers, taught the model to solve problems it couldn't before. On the test set the gain is smaller but holds (next section).

## Test Set and Possible Contamination

The 1,319-problem test set was untouched until every choice was made. Every model is evaluated in one session on one GPU ([`notebooks/04_test_eval.ipynb`](notebooks/04_test_eval.ipynb)).

| Model | Prompt | Validation (499) | Test (1,319) | Drop |
| --- | --- | --- | --- | --- |
| Base | 8-shot | 46.9% | 36.0% | −10.9 |
| Base + SFT | zero-shot | 46.5% | 35.0% | −11.5 |
| Base + SFT + GRPO (step 400) | zero-shot | 64.7% | 47.7% | −17.0 |
| Instruct | chat, zero-shot | 53.3% (61.5% lenient) | 38.8% (46.9% lenient) | −14.5 (−14.6) |

What holds on unseen problems, with paired comparisons on the same 1,319 problems:

- **SFT changed format only:** 35.0% against base 8-shot's 36.0%.
- **Qwen's post-training improved the math:** instruct's lenient 46.9% against base's 36.0% (`z` = 7.2). As scored, its lead is not clear (38.8%, `z` = 1.8): its unformatted answers hide most of the gain, as on validation.
- **GRPO added real math ability:** 35.0% to 47.7%, +12.7 points over SFT (`z` = 9.0; 258 problems only GRPO solves, 90 only SFT). About 70% of the validation gain (12.7 of 18.2 points) carries over.
- **GRPO against instruct:** clearly above it as scored (`z` = 5.7), and level with it once its format is fixed (47.7% against 46.9%, `z` = 0.6). GRPO closed the gap Stage 2 found between this model and Qwen's instruct model, but doesn't pass it. On validation, 64.7% against 61.5% looked like a lead; on test it isn't one.

**Possible contamination.** Every model scores 11-17 points lower on test than on validation. The validation subset is a random part of GSM8K train, and train and test come from the same source, so a gap this size is far beyond noise (±2.2 and ±1.4). The most likely cause is that GSM8K train problems were in Qwen2.5's training data: validation problems were familiar, test problems are not. Other signs fit:

- The base model's outputs carry instruction-tuning data from pretraining (Stage 1), so its pretraining data was not limited to raw text.
- Base and SFT run into the token limit more often on test (base 4.1% against 2.4%, SFT 3.0% against 1.2%), as if unfamiliar problems were harder to finish.
- The instruct model drops more than the base (−14.5 against −10.9), consistent with GSM8K-style data also in its post-training.

This can't be proven from accuracy alone; checking would need Qwen2.5's training data. What it changes: absolute validation scores are inflated for every model, while comparisons between models on the same problems still hold, and every validation conclusion survives on test except GRPO leading instruct's lenient score. GRPO drops the most (−17.0 against −11.5 for SFT): trained on the same train split, part of its validation gain likely came from making the model answer familiar problems more reliably, which doesn't carry over to unseen ones.

## What Specializing Cost

Baseline on the general check, before any training:

- **Base (zero-shot):** 16/20 content, 6/20 clean. It knows most answers but rarely stops.
- **Instruct:** 17/20 content, 16/20 clean. It stops reliably, which is where it differs from the base. Content is about the same.

Not stopping is what SFT should fix (EOS at the end of each completion), while content is what math specialization may erode.

After SFT:

- **SFT (zero-shot):** 8/20 content, 3/20 clean. Half the content passes are gone, mostly from the math format leaking: 7 answers contain a `####` line (`#### Tokyo`), which fails both marks. Stopping didn't improve outside math: it still appends "You are an AI assistant…", and on open-ended writing it repeats or lists until the token limit. It learned to stop after a `####` line, not after an answer.

After GRPO:

- **SFT + GRPO (step 400):** 10/20 content, 5/20 clean. Nothing was lost against SFT, and the 2-point gain is within noise on 20 prompts: a 13-point math gain on test (18 on validation) came with no further general cost. The `####` leak, SFT's cost, still affects 7 prompts.

## What Surprised Me

- The base model's appended text in the general check (an "AI assistant" system prompt, "single-select problem" blocks) is instruction-tuning data from pretraining showing through, which suggests the "base" model is not a clean base.
- The instruct model applies step-by-step reasoning even to trivia ("To determine..."), a habit its post-training carried into every kind of task.
- 800 steps of GRPO on about 6,400 train problems brought the model level with Qwen's instruct model once its format is fixed, while the 100-step smoke run, with a rising reward, changed nothing measurable. A rising training reward alone said little; only the greedy eval separated sharpening from new ability.
- Every model scored 11-17 points lower on the test set than on the validation subset taken from train, most likely because GSM8K train problems were in Qwen2.5's training data. Validation was still fine for comparing models and choosing a checkpoint, but its absolute numbers were not, and on validation GRPO seemed to beat the instruct model when on test it only matches it.

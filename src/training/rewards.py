import re

from src.evaluation.extraction import answer_extractor
from src.prompts.few_shot import STOP

FORMAT_BONUS = 0.1
FINAL_LINE = re.compile(r"^####\s*-?[\d,]*\.?\d+\s*$")

def cut_at_stop(completion: str) -> str:
    """
    Cuts a completion at the first stop string, as generate() does at eval time.
    GRPO sampling only stops at EOS or the token limit, so without the cut a made-up follow-up question
    (and its '####' line) would be scored instead of the real answer.

    Args:
        completion (str): Sampled completion.
    Returns:
        str: Completion up to the first stop string.
    """
    for s in STOP:
        completion = completion.split(s)[0]
    return completion

def correctness_reward(completions: list[str], gold: list[int], **kwargs) -> list[float]:
    """
    1.0 if the extracted answer equals the gold answer, else 0.0. Same cut and extractor as eval.

    Args:
        completions (list[str]): Sampled completions.
        gold (list[int]): Gold answers, one per completion (the dataset's "gold" column).
    Returns:
        list[float]: One reward per completion.
    """
    rewards = []
    for completion, g in zip(completions, gold):
        pred = answer_extractor(cut_at_stop(completion))
        rewards.append(1.0 if pred is not None and pred == g else 0.0)
    return rewards

def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Small bonus when the completion ends with a '#### <number>' line.

    Args:
        completions (list[str]): Sampled completions.
    Returns:
        list[float]: FORMAT_BONUS or 0.0 per completion.
    """
    rewards = []
    for completion in completions:
        last_line = cut_at_stop(completion).strip().split("\n")[-1]
        rewards.append(FORMAT_BONUS if FINAL_LINE.search(last_line) else 0.0)
    return rewards

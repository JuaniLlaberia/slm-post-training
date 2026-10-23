import re

def answer_extractor(answer: str) -> float | None:
    """
    Extracts the final number from a free-text response.
    Looks at the last '####' first (our format), then the last '\\boxed{...}' (Qwen instruct's format).

    Args:
        answer (str): Generated answer.
    Returns:
        float | None: Extracted number, or None if no final answer is found.
    """
    if "####" in answer:
        tail = answer.split("####")[-1]
    elif "\\boxed{" in answer:
        tail = answer.split("\\boxed{")[-1].split("}")[0]
    else:
        return None

    m = re.search(r"-?[\d,]*\.?\d+", tail)
    if m is None:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None

def lenient_extractor(answer: str) -> float | None:
    """
    Extracts the first number after "answer is" in a free-text response.
    Diagnostic only (never used for scoring): separates format failures from math failures.

    Args:
        answer (str): Generated answer.
    Returns:
        float | None: Extracted number, or None if no "answer is N" is found.
    """
    m = re.search(r"answer is\s*\$?(-?[\d,]*\.?\d+)", answer)
    return float(m.group(1).replace(",", "")) if m else None

import json
from collections.abc import Callable
from pathlib import Path

from src.evaluation.extraction import answer_extractor
from src.evaluation.generation import generate

def evaluate(
    model_name: str,
    prompt_builder: Callable,
    questions: list[str],
    golds: list[int],
    stop: list[str] | None = None,
    batch_size: int = 16,
    max_new_tokens: int = 512,
) -> list[dict]:
    """
    Scores one model on GSM8K-style questions.

    Args:
        model_name (str): Hugging Face model id or local path.
        prompt_builder (Callable): Turns (question, tokenizer) into the full prompt string.
        questions (list[str]): Questions to answer.
        golds (list[int]): Correct final answers, as numbers (not strings).
        stop (list[str] | None): Stop strings passed to generate.
        batch_size (int): Prompts per generate call.
        max_new_tokens (int): Generation limit per question.
    Returns:
        list[dict]: One row per question: question, output, extracted, gold, correct, n_tokens, hit_limit.
    """
    generations = generate(model_name, prompt_builder, questions, stop, batch_size, max_new_tokens)

    rows = []
    for q, g, gen in zip(questions, golds, generations):
        pred = answer_extractor(gen["output"])
        rows.append({
            "question": q,
            "output": gen["output"],
            "extracted": pred,
            "gold": g,
            "correct": pred is not None and pred == g,
            "n_tokens": gen["n_tokens"],
            "hit_limit": gen["hit_limit"],
        })

    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    no_answer = sum(r["extracted"] is None for r in rows) / n
    hit_limit = sum(r["hit_limit"] for r in rows) / n

    print(f"{model_name}: acc={acc:.1%}  no-answer={no_answer:.1%}  hit-limit={hit_limit:.1%}")
    return rows

def save_results(results: list[dict], path: str | Path) -> None:
    """
    Saves evaluation or generation rows to a JSON file, creating parent folders if needed.

    Args:
        results (list[dict]): Rows returned by evaluate or generate.
        path (str | Path): Output JSON path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

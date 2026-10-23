import re

from datasets import Dataset, load_dataset

from src.config import DATASET_CONFIG, DATASET_PATH, SEED, VAL_SIZE

def add_gold_column(example: dict) -> dict:
    """
    Adds a "gold" column with the final answer of a GSM8K solution (the number after '####').

    Args:
        example (dict): GSM8K example with an "answer" field.
    Returns:
        dict: Example with the new "gold" field.
    """
    example["gold"] = int(example["answer"].split("####")[-1].strip().replace(",", ""))
    return example

def clean_solution(example: dict) -> dict:
    """
    Removes GSM8K calculator annotations, e.g. '15*3=<<15*3=45>>45' -> '15*3=45'.

    Args:
        example (dict): GSM8K example with an "answer" field.
    Returns:
        dict: Example with the cleaned "answer" field.
    """
    example["answer"] = re.sub(r"<<[^>]*>>", "", example["answer"])
    return example

def load_gsm8k(val_size: float = VAL_SIZE, seed: int = SEED) -> tuple[Dataset, Dataset, Dataset]:
    """
    Loads GSM8K, holds out a seeded validation subset from train, and adds the gold and clean columns.

    Args:
        val_size (float): Fraction of the train split held out as validation.
        seed (int): Seed for the train/validation split.
    Returns:
        tuple[Dataset, Dataset, Dataset]: Train, validation and test datasets.
    """
    dataset = load_dataset(DATASET_PATH, DATASET_CONFIG)

    train_val_dataset = dataset["train"].train_test_split(test_size=val_size, seed=seed)
    splits = (train_val_dataset["train"], train_val_dataset["test"], dataset["test"])

    return tuple(split.map(add_gold_column).map(clean_solution) for split in splits)

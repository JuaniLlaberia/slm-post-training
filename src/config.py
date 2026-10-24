from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

DATASET_PATH = "openai/gsm8k"
DATASET_CONFIG = "main"
VAL_SIZE = 0.06675

SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

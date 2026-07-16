import json
import os
from pathlib import Path

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"
CACHE_DIR = PROJECT_ROOT / "cache"

BM25_K1 = 1.5
BM25_B = 0.75


def load_movies() -> list[dict[str, str | int]]:
    with open(DATA_PATH) as f:
        data = json.load(f)["movies"]
    return data

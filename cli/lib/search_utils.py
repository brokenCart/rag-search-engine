import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"


def load_movies() -> list[dict[str, str | int]]:
    with open(DATA_PATH) as f:
        data = json.load(f)["movies"]
    return data

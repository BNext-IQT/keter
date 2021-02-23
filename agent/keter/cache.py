import os
from pathlib import Path
from typing import Sequence
from dvc.repo import Repo

CACHE_ROOT = (
    os.environ.get("KETER_CACHE") or (Path(__file__).parent.parent / "cache").absolute()
)
DATA_ROOT = CACHE_ROOT / "data"
MODEL_ROOT = CACHE_ROOT / "models"
OUTPUTS_ROOT = CACHE_ROOT / "outputs"



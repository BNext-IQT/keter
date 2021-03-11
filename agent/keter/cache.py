import os
import pickle
from pathlib import Path
from typing import Sequence, Callable
import lzma
from dvc.repo import Repo

CACHE_ROOT = (
    os.environ.get("KETER_CACHE") or (Path(__file__).parent.parent / "cache").absolute()
)
DATA_ROOT = CACHE_ROOT / "data"
MODEL_ROOT = CACHE_ROOT / "models"
OUTPUTS_ROOT = CACHE_ROOT / "outputs"


def cache(filename: Path, func: Callable, mode="b"):
    path = filename.with_suffix(".pkz")

    if path.exists():
        with lzma.open(path, "r" + mode) as fd:
            return pickle.load(fd)
    else:
        obj = func()
        path.parents[0].mkdir(parents=True, exist_ok=True)
        with lzma.open(path, "w" + mode) as fd:
            pickle.dump(obj, fd)
        return obj

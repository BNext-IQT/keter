import os
import pickle
from pathlib import Path
from typing import Sequence, Callable, Any
import lzma
import pandas as pd
from dvc.repo import Repo


class Stage:
    CACHE_ROOT = (
        os.environ.get("KETER_CACHE")
        or (Path(__file__).parent.parent.parent / "cache").absolute()
    )

    DATA_ROOT = CACHE_ROOT / "data"
    MODEL_ROOT = CACHE_ROOT / "models"
    OUTPUTS_ROOT = CACHE_ROOT / "outputs"

    def _read_cache(self, path: Path):
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".pkz":
            with lzma.open(path, "rb") as fd:
                return pickle.load(fd)
        elif path.suffix == ".txt.xz":
            with lzma.open(seq_file, "rt") as fd:
                return fd.readlines()


class NullStage(Stage):
    def cache(self, path: Path, func: Callable, mode=None):
        return func()


class ReadOnlyStage(Stage):
    def cache(self, path: Path, func: Callable, mode=None):
        if path.exists():
            return self._read_cache(path)
        else:
            return func()


class FileSystemStage(Stage):
    def cache(self, path: Path, func: Callable, mode="b") -> Any:
        if path.exists():
            return self._read_cache(path)
        else:
            obj = func()
            path.parents[0].mkdir(parents=True, exist_ok=True)
            if isinstance(obj, pd.DataFrame):
                obj.to_parquet(path)
            else:
                with lzma.open(path, "w" + mode) as fd:
                    pickle.dump(obj, fd)
            return obj


def cache_old(files: Sequence, msg: str):
    return
    repo = Repo()
    repo.add(files)
    repo.commit(files)
    repo.push(files)

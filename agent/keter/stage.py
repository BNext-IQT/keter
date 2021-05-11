import os
from enum import Enum
import pickle
from pathlib import Path
from typing import Sequence, Callable, Any
import lzma
import pandas as pd


_stage = [None]


class Stage:
    def __init__(self):
        global _stage
        self._old_stage = _stage[0]
        _stage[0] = self

    def __enter__(self):
        pass

    def __exit__(self, *kwargs):
        global _stage
        _stage[0] = self._old_stage

    def _read_cache(self, path: Path):
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        elif path.suffix == ".pkz":
            with lzma.open(path, "rb") as fd:
                return pickle.load(fd)
        elif path.suffix == ".txt.xz":
            with lzma.open(path, "rt") as fd:
                return fd.readlines()
        else:
            with open(path, "rb") as fd:
                return fd.read()


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
            elif isinstance(obj, bytes):
                with open(path, "w" + mode) as fd:
                    fd.write(obj)
            else:
                with lzma.open(path, "w" + mode) as fd:
                    pickle.dump(obj, fd)
            return obj


def cache(product: str, name: str, func: Callable) -> Any:
    stage = _stage[0]
    if not stage:
        raise ValueError(
            "There is no stage defined. Make sure you only call "
            "actors and datasets inside of a stage block."
        )
    else:
        return stage.cache(get_path(product) / name, func)


def get_path(product: str) -> Path:
    CACHE_ROOT = (
        os.environ.get("KETER_CACHE")
        or (Path(__file__).parent.parent.parent / "cache").absolute()
    )
    if product == "root":
        return CACHE_ROOT
    elif product == "raw":
        return CACHE_ROOT / "data" / "raw"
    elif product == "constructed":
        return CACHE_ROOT / "data" / "constructed"
    elif product == "model":
        return CACHE_ROOT / "models"
    elif product == "output":
        return CACHE_ROOT / "outputs"
    elif product == "external":
        return CACHE_ROOT / "external"
    else:
        raise ValueError(f"Invalid product: {product}")

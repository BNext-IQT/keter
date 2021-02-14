import pandas as pd
import numpy as np
from keter.cache import CACHE_ROOT
from keter.datasets.raw import Tox21, ToxCast

CONSTRUCTED_DATA_PATH = CACHE_ROOT / "data" / "constructed"
CONSTRUCTED_DATA_PATH.mkdir(parents=True, exist_ok=True)


class ConstructedData:
    def __call__(self, override=False) -> pd.DataFrame:
        parquet_file = (CONSTRUCTED_DATA_PATH / self.filename).with_suffix(".parquet")
        if parquet_file.exists() and not override:
            dataframe = pd.read_parquet(parquet_file)
        else:
            dataframe = self.construct()
            dataframe.to_parquet(parquet_file)
        return dataframe


class Toxicity(ConstructedData):
    filename = "toxicity"

    def construct(self) -> pd.DataFrame:
        dataframe = pd.merge(
            self._normalize_tox(Tox21()()),
            self._normalize_tox(ToxCast()()),
            on="smiles",
            how="outer",
            sort=False,
        )
        dataframe["toxicity"] = dataframe[["toxicity_x", "toxicity_y"]].mean(axis=1)
        dataframe = dataframe.drop(["toxicity_x", "toxicity_y"], axis=1)
        return dataframe.reset_index(drop=True).copy()

    def _normalize_tox(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["toxicity"] = dataframe.sum(axis=1)

        # Normalize toxicity score
        max_val = dataframe["toxicity"].max()
        dataframe["toxicity"] = dataframe["toxicity"] ** (1 / 3) / np.cbrt(max_val)

        return dataframe[["smiles", "toxicity"]]
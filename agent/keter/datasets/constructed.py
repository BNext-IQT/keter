import pandas as pd
import numpy as np
from keter.cache import DATA_ROOT
from keter.datasets.raw import Tox21, ToxCast, Moses, Bbbp, Muv, ClinTox, Pcba, Sider

CONSTRUCTED_DATA_ROOT = DATA_ROOT / "constructed"


class ConstructedData:
    def to_df(self, cache=False) -> pd.DataFrame:
        parquet_file = (CONSTRUCTED_DATA_ROOT / self.filename).with_suffix(".parquet")
        if parquet_file.exists():
            dataframe = pd.read_parquet(parquet_file)
        else:
            dataframe = self.construct(cache)
            if cache:
                CONSTRUCTED_DATA_ROOT.mkdir(parents=True, exist_ok=True)
                dataframe.to_parquet(parquet_file)
        return dataframe


class Toxicity(ConstructedData):
    filename = "toxicity"

    def construct(self, cache) -> pd.DataFrame:
        dataframe = pd.merge(
            self._normalize_tox(Tox21().to_df(cache)),
            self._normalize_tox(ToxCast().to_df(cache)),
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


class Unlabeled(ConstructedData):
    filename = "unlabeled"

    def construct(self, cache) -> pd.DataFrame:
        dataframe = (
            pd.concat(
                [
                    Moses()
                    .to_df(cache)[["SMILES"]]
                    .rename(columns={"SMILES": "smiles"}),
                    ToxCast().to_df(cache)[["smiles"]],
                    Tox21().to_df(cache)[["smiles"]],
                    Bbbp().to_df(cache)[["smiles"]],
                    Muv().to_df(cache)[["smiles"]],
                    Sider().to_df(cache)[["smiles"]],
                    ClinTox().to_df(cache)[["smiles"]],
                    Pcba().to_df(cache)[["smiles"]],
                ]
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return dataframe
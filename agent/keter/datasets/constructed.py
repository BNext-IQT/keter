import lzma
from typing import List, Sequence
from dateutil.parser import parse
import pandas as pd
import numpy as np
from keter.cache import DATA_ROOT
from keter.datasets.raw import (
    Tox21,
    ToxCast,
    Moses,
    Bbbp,
    Muv,
    ClinTox,
    Pcba,
    Sider,
    CoronaDeathsUSA,
)
from keter.operations import construct_infection_records

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

    def construct(self, cache: bool) -> pd.DataFrame:
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

    def to_list(self, cache=False) -> List[str]:
        seq_file = (CONSTRUCTED_DATA_ROOT / self.filename).with_suffix(".txt.xz")
        if seq_file.exists():
            with lzma.open(seq_file, "rt") as fd:
                return fd.readlines()
        else:
            if cache:
                fd = lzma.open(seq_file, "wt")
            res = []
            for smiles in self.to_df(cache=False).squeeze():
                if cache:
                    fd.write(smiles + "\n")
                res.append(smiles)
            if cache:
                fd.close()
            return res

    def construct(self, cache: bool) -> pd.DataFrame:
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


class InfectionNet:
    filename = "infectionnet"

    def to_csv(self, cache=False) -> Sequence[str]:
        csv_file = (CONSTRUCTED_DATA_ROOT / self.filename).with_suffix(".csv.xz")

        if csv_file.exists():
            with lzma.open(csv_file, "rt") as fd:
                for line in fd:
                    yield line.rstrip()
                return

        corona_deaths = CoronaDeathsUSA().to_df(cache)
        corona_deaths = corona_deaths.rename(
            columns={
                column: int(parse(column).timestamp())
                for column in corona_deaths.columns
                if "/" in column
            }
        )
        timestamp_columns = [
            column for column in corona_deaths.columns if isinstance(column, int)
        ]
        corona_deaths[timestamp_columns] = corona_deaths[timestamp_columns].diff(axis=1)
        corona_deaths = corona_deaths.dropna(axis=1)

        if cache:
            CONSTRUCTED_DATA_ROOT.mkdir(parents=True, exist_ok=True)
            fd = lzma.open(csv_file, "wt")
        for row in corona_deaths.iterrows():
            _, series = row
            for column, val in series.items():
                if isinstance(column, int):
                    for record in construct_infection_records(
                        column, val, series.Lat, series.Long_
                    ):
                        if cache:
                            fd.write(record + "\n")
                        yield record
        if cache:
            fd.close()
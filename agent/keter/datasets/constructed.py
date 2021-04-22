import lzma
from typing import List, Sequence
from dateutil.parser import parse
import pandas as pd
import numpy as np
from keter.stage import cache, get_path
from keter.datasets.raw import (
    Tox21,
    ToxCast,
    Moses,
    Bbbp,
    Muv,
    ClinTox,
    Pcba,
    Sider,
    Bbbp,
    Lipophilicity,
    CoronaDeathsUSA,
    ESOL,
)
from keter.operations import construct_infection_records


class ConstructedData:
    def to_df(self) -> pd.DataFrame:
        name = self.filename + ".parquet"
        return cache("constructed", name, lambda: self.construct())


class Safety(ConstructedData):
    filename = "safety"

    def construct(self) -> pd.DataFrame:
        dataframe = pd.merge(
            self._normalize_tox(Tox21().to_df()),
            self._normalize_tox(ToxCast().to_df()),
            on="smiles",
            how="outer",
            sort=False,
        )
        dataframe["safety"] = dataframe[["safety_x", "safety_y"]].mean(axis=1)
        dataframe = dataframe.drop(["safety_x", "safety_y"], axis=1)
        return dataframe.reset_index(drop=True).copy()

    def _normalize_tox(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["safety"] = (
            dataframe.replace(to_replace=0.0, value=-1.0).fillna(0.0).sum(axis=1)
        )

        # Normalize safety score
        min_val = dataframe["safety"].min()
        max_val = dataframe["safety"].max()
        dataframe["safety"] = 1 - (dataframe["safety"] - min_val) ** (1 / 2) / np.sqrt(
            np.abs(min_val) + max_val
        )

        return dataframe[["smiles", "safety"]]


class Feasibility(ConstructedData):
    filename = "feasibility"

    def construct(self) -> pd.DataFrame:
        esol = ESOL().to_df()
        esol_ylabel = "ESOL predicted log solubility in mols per litre"

        # Normalize feasibility score
        min_val = esol[esol_ylabel].min()
        max_val = esol[esol_ylabel].max()
        esol["feasibility"] = (esol[esol_ylabel] - min_val) / (
            np.abs(min_val) + max_val
        )

        return esol[["smiles", "feasibility"]].reset_index(drop=True).copy()


class Unlabeled(ConstructedData):
    filename = "unlabeled"

    def to_list(self) -> List[str]:
        return self.to_df()["smiles"].tolist()

    def construct(self) -> pd.DataFrame:
        dataframe = (
            pd.concat(
                [
                    Moses().to_df()[["SMILES"]].rename(columns={"SMILES": "smiles"}),
                    ToxCast().to_df()[["smiles"]],
                    Tox21().to_df()[["smiles"]],
                    Bbbp().to_df()[["smiles"]],
                    Muv().to_df()[["smiles"]],
                    Sider().to_df()[["smiles"]],
                    ClinTox().to_df()[["smiles"]],
                    Pcba().to_df()[["smiles"]],
                    Bbbp().to_df()[["smiles"]],
                    Lipophilicity().to_df()[["smiles"]],
                ]
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return dataframe


class InfectionNet:
    filename = "infectionnet"

    # TODO: Respect the stage
    def to_csv(self) -> Sequence[str]:
        constructed_data_root = get_path("constructed")
        csv_file = (constructed_data_root / self.filename).with_suffix(".csv.xz")

        if csv_file.exists():
            with lzma.open(csv_file, "rt") as fd:
                for line in fd:
                    yield line.rstrip()
                return

        corona_deaths = CoronaDeathsUSA().to_df()
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

        constructed_data_root.mkdir(parents=True, exist_ok=True)
        fd = lzma.open(csv_file, "wt")
        for row in corona_deaths.iterrows():
            _, series = row
            for column, val in series.items():
                if isinstance(column, int):
                    for record in construct_infection_records(
                        column, val, series.Lat, series.Long_
                    ):
                        fd.write(record + "\n")
                        yield record
        fd.close()
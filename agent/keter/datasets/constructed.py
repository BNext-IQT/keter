import lzma
from typing import List, Sequence
from dateutil.parser import parse
import pandas as pd
import numpy as np
from keter.stage import Stage, ReadOnlyStage
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
    ESOL,
)
from keter.operations import construct_infection_records


class ConstructedData:
    def to_df(self, stage: Stage = ReadOnlyStage()) -> pd.DataFrame:
        path = (stage.DATA_ROOT / "constructed" / self.filename).with_suffix(".parquet")
        return stage.cache(path, lambda: self.construct(stage))


class Toxicity(ConstructedData):
    filename = "toxicity"

    def construct(self, stage: Stage) -> pd.DataFrame:
        dataframe = pd.merge(
            self._normalize_tox(Tox21().to_df(stage)),
            self._normalize_tox(ToxCast().to_df(stage)),
            on="smiles",
            how="outer",
            sort=False,
        )
        dataframe["toxicity"] = dataframe[["toxicity_x", "toxicity_y"]].mean(axis=1)
        dataframe = dataframe.drop(["toxicity_x", "toxicity_y"], axis=1)
        return dataframe.reset_index(drop=True).copy()

    def _normalize_tox(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe["toxicity"] = (
            dataframe.replace(to_replace=0.0, value=-1.0).fillna(0.0).sum(axis=1)
        )

        # Normalize toxicity score
        min_val = dataframe["toxicity"].min()
        max_val = dataframe["toxicity"].max()
        dataframe["toxicity"] = (dataframe["toxicity"] - min_val) ** (1 / 2) / np.sqrt(
            np.abs(min_val) + max_val
        )

        return dataframe[["smiles", "toxicity"]]


class Feasibility(ConstructedData):
    filename = "feasibility"

    def construct(self, stage: Stage = ReadOnlyStage()) -> pd.DataFrame:
        esol = ESOL().to_df(stage)
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

    # TODO: Respect the stage
    def to_list(self, stage: Stage = ReadOnlyStage()) -> List[str]:
        path = (stage.DATA_ROOT / "constructed" / self.filename).with_suffix(".txt.xz")
        if path.exists():
            with lzma.open(seq_file, "rt") as fd:
                return fd.readlines()
        else:
            fd = lzma.open(seq_file, "wt")
            res = []
            for smiles in self.to_df(stage=False).squeeze():
                fd.write(smiles + "\n")
                res.append(smiles)
                fd.close()
            return res

    def construct(self, stage: Stage) -> pd.DataFrame:
        dataframe = (
            pd.concat(
                [
                    Moses()
                    .to_df(stage)[["SMILES"]]
                    .rename(columns={"SMILES": "smiles"}),
                    ToxCast().to_df(stage)[["smiles"]],
                    Tox21().to_df(stage)[["smiles"]],
                    Bbbp().to_df(stage)[["smiles"]],
                    Muv().to_df(stage)[["smiles"]],
                    Sider().to_df(stage)[["smiles"]],
                    ClinTox().to_df(stage)[["smiles"]],
                    Pcba().to_df(stage)[["smiles"]],
                ]
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )
        return dataframe


class InfectionNet:
    filename = "infectionnet"

    # TODO: Respect the stage
    def to_csv(self, stage: Stage = ReadOnlyStage()) -> Sequence[str]:
        constructed_data_root = stage.DATA_ROOT / "constructed"
        csv_file = (constructed_data_root / self.filename).with_suffix(".csv.xz")

        if csv_file.exists():
            with lzma.open(csv_file, "rt") as fd:
                for line in fd:
                    yield line.rstrip()
                return

        corona_deaths = CoronaDeathsUSA().to_df(stage)
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
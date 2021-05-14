from typing import Sequence
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
from functools import reduce
from urllib.request import urlopen
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keter.actors.vectors import ChemicalLanguage
from keter.stage import get_path, cache
from keter.datasets.constructed import ConstructedData


class Tox21Full(ConstructedData):
    filename = "tox21_full_combined"
    tox21_assays = [
        "tox21-ahr-p1",
        "tox21-ap1-agonist-p1",
        "tox21-ar-bla-agonist-p1",
        "tox21-ar-bla-antagonist-p1",
        "tox21-ar-mda-kb2-luc-agonist-p1",
        "tox21-ar-mda-kb2-luc-agonist-p3",
        "tox21-ar-mda-kb2-luc-antagonist-p1",
        "tox21-ar-mda-kb2-luc-antagonist-p2",
        "tox21-are-bla-p1",
        "tox21-aromatase-p1",
        "tox21-car-agonist-p1",
        "tox21-car-antagonist-p1",
        "tox21-casp3-cho-p1",
        "tox21-casp3-hepg2-p1",
        "tox21-dt40-p1",
        "tox21-elg1-luc-agonist-p1",
        "tox21-er-bla-agonist-p2",
        "tox21-er-bla-antagonist-p1",
        "tox21-er-luc-bg1-4e2-agonist-p2",
        "tox21-er-luc-bg1-4e2-agonist-p4",
        "tox21-er-luc-bg1-4e2-antagonist-p1",
        "tox21-er-luc-bg1-4e2-antagonist-p2",
        "tox21-erb-bla-antagonist-p1",
        "tox21-erb-bla-p1",
        "tox21-err-p1",
        "tox21-esre-bla-p1",
        "tox21-fxr-bla-agonist-p2",
        "tox21-fxr-bla-antagonist-p1",
        "tox21-gh3-tre-agonist-p1",
        "tox21-gh3-tre-antagonist-p1",
        "tox21-gr-hela-bla-agonist-p1",
        "tox21-gr-hela-bla-antagonist-p1",
        "tox21-h2ax-cho-p2",
        "tox21-hdac-p1",
        "tox21-hre-bla-agonist-p1",
        "tox21-hse-bla-p1",
        "tox21-luc-biochem-p1",
        "tox21-mitotox-p1",
        "tox21-nfkb-bla-agonist-p1",
        "tox21-p53-bla-p1",
        "tox21-pgc-err-p1",
        "tox21-ppard-bla-agonist-p1",
        "tox21-ppard-bla-antagonist-p1",
        "tox21-pparg-bla-agonist-p1",
        "tox21-pparg-bla-antagonist-p1",
        "tox21-pr-bla-agonist-p1",
        "tox21-pr-bla-antagonist-p1",
        "tox21-pxr-p1",
        "tox21-rar-agonist-p1",
        "tox21-rar-antagonist-p2",
        "tox21-ror-cho-antagonist-p1",
        "tox21-rt-viability-hek293-p1",
        "tox21-rt-viability-hepg2-p1",
        "tox21-rxr-bla-agonist-p1",
        "tox21-sbe-bla-agonist-p1",
        "tox21-sbe-bla-antagonist-p1",
        "tox21-shh-3t3-gli3-agonist-p1",
        "tox21-shh-3t3-gli3-antagonist-p1",
        "tox21-trhr-hek293-p1",
        "tox21-tshr-agonist-p1",
        "tox21-tshr-antagonist-p1",
        "tox21-tshr-wt-p1",
        "tox21-vdr-bla-agonist-p1",
        "tox21-vdr-bla-antagonist-p1",
    ]

    def download(self, assay: str):
        if assay not in self.tox21_assays:
            raise ValueError(f"Not a valid Tox21 assay: {assay}")
        raw_dir = get_path("raw") / "tox21"
        raw_url = f"https://tripod.nih.gov/tox21/assays/download/{assay}.zip"
        return urlopen(raw_url).read()

    def to_df_by_assay(self, assay: str) -> pd.DataFrame:
        raw = cache("raw", Path("tox21") / f"{assay}.zip", lambda: self.download(assay))
        raw_fd = BytesIO(raw)
        with ZipFile(raw_fd) as zip_fd:
            for inner_filename in zip_fd.namelist():
                if inner_filename.endswith("aggregrated.txt"):
                    with zip_fd.open(inner_filename) as inner_fd:
                        return pd.read_csv(inner_fd, sep="\t", index_col=False)

    def to_dfs(self) -> Sequence[pd.DataFrame]:
        for assay in self.tox21_assays:
            yield assay, self.to_df_by_assay(assay)

    def construct(self) -> pd.DataFrame:
        def create_assay_df(assay: str, df: pd.DataFrame) -> pd.DataFrame:
            if "antagonist" in assay:
                test_type = "active antagonist"
            elif "agonist" in assay:
                test_type = "active agonist"
            else:
                test_type = "active "

            def clarify_df(raw_df: pd.DataFrame) -> pd.DataFrame:
                for name, group in raw_df.groupby("SMILES"):
                    try:
                        if group["ASSAY_OUTCOME"].str.contains(test_type).any():
                            yield name, 1.0
                        else:
                            yield name, 0.0
                    except AttributeError:
                        yield name, float("NaN")

            return pd.DataFrame(clarify_df(df), columns=["smiles", assay])

        result = reduce(
            lambda left, right: pd.merge(
                left, right, on="smiles", how="outer", sort=False,
            ),
            (
                create_assay_df(assay, df)
                for assay, df in tqdm(
                    self.to_dfs(),
                    total=len(self.tox21_assays),
                    unit="assay",
                    desc="[Dataset] Tox21Full",
                )
            ),
        )

        return result


class Safety(ConstructedData):
    filename = "safety2"

    def __init__(self):
        self.preprocessor = ChemicalLanguage("bow")

    def _determine_assay_score(self, X: pd.Series, y: pd.Series) -> float:
        Xt, Xv, yt, yv = train_test_split(
            self.preprocessor.transform(X),
            y,
            test_size=0.15,
            random_state=18,
            stratify=y,
        )

        model = RandomForestClassifier()
        model.fit(Xt, yt)
        yhat = model.predict_proba(Xv)
        score = roc_auc_score(yv, yhat, multi_class="ovr")
        return max(0.0, (score - 0.5) * 2)

    def construct(self) -> dict:
        tox21 = Tox21Full()
        df = tox21.to_df()
        X = df["smiles"]
        df = df.replace([float("NaN"), 1.0, 0.0], [0.0, 1.0, -1.0])
        for column in tqdm(
            df,
            total=len(tox21.tox21_assays) + 1,
            unit="assay",
            desc="[Dataset] Safety",
        ):
            if column == "smiles":
                continue
            y = df[column]
            try:
                score = self._determine_assay_score(X, y)
                df[column] = df[column] * score
            except:
                df = df.drop(columns=column)

        df["safety"] = df.sum(axis=1)
        min_val = df["safety"].min()
        max_val = df["safety"].max()
        df["safety"] = 1 - (df["safety"] - min_val) ** (1 / 2) / np.sqrt(
            np.abs(min_val) + max_val
        )

        return df[["smiles", "safety"]].reset_index(drop=True)

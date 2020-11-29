from typing import Generator, Sequence
import gzip
from uuid import uuid4
import pandas as pd
import deepchem as dc
import selfies
from transformers import Trainer, TrainingArguments


def df2corpus(df: pd.DataFrame) -> Generator:
    if "smiles" in df:
        df.dropna(subset=["smiles"], inplace=True)
        df["smiles"] = df["smiles"].apply(selfies.encoder)

class Beresheet:
    def fit(self, dataset: str):
        with gzip.open(dataset, "rb") as sentences:
            for sentence in sentences:




        



class DeepChem:
    gcm_model_version = "1"

    def __init__(self, path):
        self.molnet_dir = path / ".." / "data" / "molnet"
        self.gcm_model_dir = (
            path / "chemistry" / "gcm" / self.gcm_model_version / str(uuid4())
        )

    def gather_data(self, task="muv"):
        if task == "tox":
            t, d, r = dc.molnet.load_tox21(split="stratified", save_dir=self.molnet_dir)
        elif task == "muv":
            t, d, r = dc.molnet.load_muv(split="stratified", save_dir=self.molnet_dir)
        self.tasks, self.datasets, self.transformers = t, d, r
        self.train, self.verify, self.test = self.datasets

    def fit(self):
        if not hasattr(self, "train"):
            self.gather_data()
        n_features = self.train.get_data_shape()[0]
        n_tasks = len(self.tasks)
        model = dc.models.MultitaskClassifier(
            n_tasks, n_features, model_dir=self.gcm_model_dir
        )
        model.fit(self.train)

        self.model = model

    def predict(self, smiles: Sequence[str]) -> pd.DataFrame:
        for mol in smiles:
            self.model.predict(mol)

    def score(self):
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        test_metric = self.model.evaluate(self.test, [metric], self.transformers)[
            "roc_auc_score"
        ]
        train_metric = self.model.evaluate(self.train, [metric], self.transformers)[
            "roc_auc_score"
        ]
        return test_metric, train_metric

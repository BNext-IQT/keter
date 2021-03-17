from typing import Sequence
from autosklearn.estimators import AutoSklearnRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keter.datasets.constructed import Toxicity
from keter.datasets.raw import Tox21
from keter.actors.vectors import ChemicalLanguage
from keter.cache import cache, MODEL_ROOT


class Analyzer:
    filename = "analyzer"

    def __init__(self):
        self.preprocessor = ChemicalLanguage("bow")
        self.model = cache(MODEL_ROOT / self.filename, self.train)

    def train(self):
        data = Toxicity().to_df()
        model = AutoSklearnRegressor(time_left_for_this_task=18000)

        X = self.preprocessor.transform(data["smiles"])
        y = data["toxicity"]

        model.fit(X, y)

        return model

    def analyze(self, smiles: Sequence[str]) -> Sequence[float]:
        return self.model.predict(self.preprocessor.transform(smiles))


class RandomForestBenchmarks:
    filename = "benchmarks_rf"

    def __init__(self):
        self.preprocessor = ChemicalLanguage("bow")
        self.model = cache(MODEL_ROOT / self.filename, self.train)

    def train(self):
        data = Tox21().to_df().fillna(-1)
        model = RandomForestClassifier()

        Xt, Xv, yt, yv = train_test_split(
            self.preprocessor.transform(data["smiles"]),
            data.drop(columns=["smiles", "mol_id"]),
            test_size=0.2,
            random_state=18,
        )

        model.fit(Xt, yt)

        return model

    def analyze(self, smiles: Sequence[str]) -> Sequence[float]:
        return self.model.predict(self.preprocessor.transform(smiles))
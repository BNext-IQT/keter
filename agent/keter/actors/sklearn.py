from typing import Sequence
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keter.datasets.constructed import Toxicity
from keter.datasets.raw import Tox21
from keter.actors.vectors import ChemicalLanguage
from keter.cache import cache, MODEL_ROOT


class RandomForestAnalyzer:
    filename = "analyzer_rf"

    def __init__(self):
        self.preprocessor = ChemicalLanguage("bow")
        self.model = cache(MODEL_ROOT / self.filename, self.train)

    def train(self):
        data = Toxicity().to_df()
        model = RandomForestRegressor()

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
            data["smiles"], data["NR-AR"], test_size=0.2, random_state=18
        )

        Xt = self.preprocessor.transform(Xt)
        Xv = self.preprocessor.transform(Xv)
        yt = yt.to_numpy()
        yv = yv.to_numpy()

        model.fit(Xt, yt)
        y_hats = model.predict_proba(Xv)

        score = roc_auc_score(yv, y_hats, multi_class="ovo")

        return model

    def analyze(self, smiles: Sequence[str]) -> Sequence[float]:
        return self.model.predict(self.preprocessor.transform(smiles))
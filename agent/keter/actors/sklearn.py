from typing import Sequence
from sklearn.ensemble import RandomForestRegressor
from keter.datasets.constructed import Toxicity
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
from typing import Sequence
from sklearn.ensemble import RandomForestRegressor
from keter.datasets.constructed import Toxicity
from keter.actors.vectors import ChemicalLanguage


class RandomForestAnalyzer:
    def __init__(self):
        self.model = self.train()

    def train(self):
        data = Toxicity().to_df()
        self.preprocessor = ChemicalLanguage("bow")
        model = RandomForestRegressor()

        X = self.preprocessor.transform(data["smiles"])
        y = data["toxicity"]

        model.fit(X, y)

        return model

    def analyze(self, smiles: Sequence[str]) -> Sequence[float]:
        return self.model.predict(self.preprocessor.transform(smiles))
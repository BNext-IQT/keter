from keter.models.vectors import ChemicalLanguage, ChemicalLanguageHyperparameters
from keter.datasets.constructed import Unlabeled, Toxicity


class ChemicalLanguageDefault:
    def __init__(self):
        self.train()

    def train(self):
        tox = Toxicity()()
        X = tox["smiles"]
        y = tox.drop("smiles", axis=1)
        y["toxicity"] = tox.apply(lambda x: 1 if x.toxicity > 0.18 else 0, axis=1)
        self.model = ChemicalLanguage(
            ChemicalLanguageHyperparameters.from_dict({"vector_algo": "bow"})
        )
        self.model.fit(Unlabeled()().squeeze(), X, y)
import pickle
from keter.models.vectors import ChemicalLanguageModule, ChemicalLanguageHyperparameters
from keter.datasets.constructed import Unlabeled, Toxicity


class ChemicalLanguage:
    def __init__(self, mode="default"):
        if mode == "default":
            self.train()
        elif mode == "bow":
            self.train(
                ChemicalLanguageHyperparameters.from_dict({"vector_algo": "bow"})
            )
        with open("ChemicalLanguage." + mode + ".pickle") as fd:
            pickle.dump(self.model, fd)

    def train(self, hyperparams=ChemicalLanguageHyperparameters()):
        tox = Toxicity().to_df(cache=True)
        X = tox["smiles"]
        y = tox.drop("smiles", axis=1)
        y["toxicity"] = tox.apply(lambda x: 1 if x.toxicity > 0.18 else 0, axis=1)
        self.model = ChemicalLanguageModule(hyperparams)
        self.model.fit(Unlabeled().to_list(cache=True), X, y)

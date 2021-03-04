import pickle
import lzma
from keter.models.vectors import ChemicalLanguageModule, ChemicalLanguageHyperparameters
from keter.datasets.constructed import Unlabeled, Toxicity
from keter.cache import MODEL_ROOT


class ChemicalLanguage:
    filename = "ChemicalLanguage"

    def __init__(self, mode="default"):
        model_file = (MODEL_ROOT / self.filename).with_suffix(f".{mode}.pickle.xz")
        if model_file.exists():
            with lzma.open(model_file, "rb") as fd:
                self.model = pickle.load(fd)
            return

        if mode == "default":
            self.model = self.train()
        elif mode == "bow":
            self.model = self.train(
                ChemicalLanguageHyperparameters.from_dict({"vector_algo": "bow"})
            )

        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        with lzma.open(model_file, "wb") as fd:
            pickle.dump(self.model, fd)

    def train(self, hyperparams=ChemicalLanguageHyperparameters()):
        tox = Toxicity().to_df(cache=True)
        X = tox["smiles"]
        y = tox.drop("smiles", axis=1)
        y["toxicity"] = tox.apply(lambda x: 1 if x.toxicity > 0.18 else 0, axis=1)
        model = ChemicalLanguageModule(hyperparams)
        model.fit(Unlabeled().to_list(cache=True), X, y)
        return model

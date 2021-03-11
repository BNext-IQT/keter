import pickle
import lzma
from typing import Sequence
from keter.models.vectors import ChemicalLanguageModule, ChemicalLanguageHyperparameters
from keter.datasets.constructed import Unlabeled, Toxicity
from keter.cache import MODEL_ROOT, cache


class ChemicalLanguage:
    filename = "chemical_language"

    def __init__(self, mode="default"):
        model_file = MODEL_ROOT / f"{self.filename}_{mode}"

        if mode == "default":
            self.model = cache(model_file, self.train)
        elif mode == "bow":
            self.model = cache(
                model_file,
                lambda: self.train(
                    ChemicalLanguageHyperparameters.from_dict({"vector_algo": "bow"})
                ),
            )
        elif mode == "lda":
            self.model = cache(
                model_file,
                lambda: self.train(
                    ChemicalLanguageHyperparameters.from_dict({"vector_algo": "lda"})
                ),
            )
        elif mode == "fastd2v":
            self.model = cache(
                model_file,
                lambda: self.train(
                    ChemicalLanguageHyperparameters.from_dict({"doc_epochs": 5})
                ),
            )
        else:
            raise ValueError("Invalid mode: " + mode)

    def train(self, hyperparams=ChemicalLanguageHyperparameters()):
        tox = Toxicity().to_df(cache=True)
        X = tox["smiles"]
        y = tox.drop("smiles", axis=1)
        y["toxicity"] = tox.apply(lambda x: 1 if x.toxicity > 0.18 else 0, axis=1)
        model = ChemicalLanguageModule(hyperparams)
        model.fit(Unlabeled().to_list(cache=True), X, y)
        return model

    def transform(self, smiles: Sequence[str]) -> Sequence[str]:
        return self.model.to_vecs(smiles)

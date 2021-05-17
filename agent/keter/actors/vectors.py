import pickle
import lzma
from typing import Sequence
from keter.models.vectors import ChemicalLanguageModule, ChemicalLanguageHyperparameters
from keter.datasets.constructed import Unlabeled, Safety
from keter.stage import cache


class ChemicalLanguage:
    filename = "chemical_language"

    def __init__(self, mode="bow"):
        model_file = f"{self.filename}_{mode}.pkz"

        if mode == "default":
            self.model = cache("model", model_file, self.train)
        elif mode == "bow":
            self.model = cache(
                "model",
                model_file,
                lambda: self.train(
                    ChemicalLanguageHyperparameters.from_dict(
                        {"vector_algo": "bow", "max_vocab": 5000, "max_ngram": 4}
                    )
                ),
            )
        elif mode == "lda":
            self.model = cache(
                "model",
                model_file,
                lambda: self.train(
                    ChemicalLanguageHyperparameters.from_dict(
                        {"vector_algo": "lda", "topics": 1000}
                    )
                ),
            )
        elif mode == "doc2vec":
            self.model = cache(
                "model",
                model_file,
                lambda: self.train(
                    ChemicalLanguageHyperparameters.from_dict(
                        {"doc_epochs": 300, "vec_dims": 512}
                    )
                ),
            )
        else:
            raise ValueError("Invalid mode: " + mode)

    def train(self, hyperparams=ChemicalLanguageHyperparameters()):
        safety = Safety().to_df()
        X = safety["smiles"]
        y = safety.drop("smiles", axis=1)
        y["safety"] = safety.apply(lambda x: 1 if x.safety > 0.7 else 0, axis=1)
        model = ChemicalLanguageModule(hyperparams)
        model.fit(Unlabeled().to_list(), X, y)
        return model

    def transform(self, smiles: Sequence[str]) -> Sequence[str]:
        return self.model.to_vecs(smiles)

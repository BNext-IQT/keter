import pickle
import lzma
from typing import Sequence
from keter.models.vectors import ChemicalLanguageModule, ChemicalLanguageHyperparameters
from keter.datasets.constructed import Unlabeled, Safety
from keter.stage import Stage, ReadOnlyStage


class ChemicalLanguage:
    filename = "chemical_language"

    def __init__(self, mode="default", stage: Stage = ReadOnlyStage()):
        model_file = (stage.MODEL_ROOT / f"{self.filename}_{mode}").with_suffix(".pkz")

        if mode == "default":
            self.model = stage.cache(model_file, self.train)
        elif mode == "bow":
            self.model = stage.cache(
                model_file,
                lambda: self.train(
                    ChemicalLanguageHyperparameters.from_dict(
                        {"vector_algo": "bow", "max_vocab": 5000, "max_ngram": 4}
                    )
                ),
            )
        elif mode == "lda":
            self.model = stage.cache(
                model_file,
                lambda: self.train(
                    ChemicalLanguageHyperparameters.from_dict({"vector_algo": "lda"})
                ),
            )
        elif mode == "fastd2v":
            self.model = stage.cache(
                model_file,
                lambda: self.train(
                    ChemicalLanguageHyperparameters.from_dict({"doc_epochs": 5})
                ),
            )
        else:
            raise ValueError("Invalid mode: " + mode)

    def train(self, hyperparams=ChemicalLanguageHyperparameters()):
        safety = Safety().to_df(cache=True)
        X = safety["smiles"]
        y = safety.drop("smiles", axis=1)
        y["safety"] = tox.apply(lambda x: 1 if x.safety > 0.7 else 0, axis=1)
        model = ChemicalLanguageModule(hyperparams)
        model.fit(Unlabeled().to_list(cache=True), X, y)
        return model

    def transform(self, smiles: Sequence[str]) -> Sequence[str]:
        return self.model.to_vecs(smiles)

import pickle
import lzma
from typing import Sequence
from keter.models.vectors import ChemicalLanguageModule, ChemicalLanguageHyperparameters
from keter.datasets.constructed import Unlabeled, Safety
from keter.stage import Stage, ReadOnlyStage


class ChemicalLanguage:
    filename = "chemical_language"

    def __init__(self, mode="bow", stage: Stage = ReadOnlyStage()):
        self.stage = stage
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
                    ChemicalLanguageHyperparameters.from_dict(
                        {"vector_algo": "lda", "topics": 100}
                    )
                ),
            )
        elif mode == "doc2vec":
            self.model = stage.cache(
                model_file,
                lambda: self.train(
                    ChemicalLanguageHyperparameters.from_dict({"doc_epochs": 30})
                ),
            )
        else:
            raise ValueError("Invalid mode: " + mode)

    def train(self, hyperparams=ChemicalLanguageHyperparameters()):
        safety = Safety().to_df(self.stage)
        X = safety["smiles"]
        y = safety.drop("smiles", axis=1)
        y["safety"] = safety.apply(lambda x: 1 if x.safety > 0.7 else 0, axis=1)
        model = ChemicalLanguageModule(hyperparams)
        model.fit(Unlabeled().to_list(stage=self.stage), X, y)
        return model

    def transform(self, smiles: Sequence[str]) -> Sequence[str]:
        return self.model.to_vecs(smiles)

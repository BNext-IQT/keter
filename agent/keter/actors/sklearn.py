from typing import Sequence
from autosklearn.estimators import AutoSklearnRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keter.datasets.constructed import Toxicity, Feasibility
from keter.datasets.raw import Tox21
from keter.actors.vectors import ChemicalLanguage
from keter.cache import cache, MODEL_ROOT


class Analyzer:
    filename = "analyzer"

    def __init__(self, mode="prod"):
        model_file = MODEL_ROOT / f"{self.filename}_{mode}"
        self.preprocessor = ChemicalLanguage("bow")
        if mode == "prod":
            self.safety_model, self._model = cache(model_file, self.train)
        elif mode == "test":
            self.safety_model, self.synth_model = self.train(
                score=True, task_duration=300
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def train(self, score=False, task_duration=14400):
        def train_model(data, target_label):
            dataframe = data.to_df(cache=True)
            model = AutoSklearnRegressor(time_left_for_this_task=task_duration)

            if score:
                Xt, Xv, yt, yv = train_test_split(
                    self.preprocessor.transform(dataframe["smiles"]),
                    dataframe[target_label],
                    test_size=0.15,
                    random_state=18,
                )
            else:
                Xt = self.preprocessor.transform(dataframe["smiles"])
                yt = dataframe[target_label]

            model.fit(Xt, yt)

            if score:
                print(f"Score on {target_label}: {model.score(Xv, yv)}")

            return model

        return train_model(Toxicity(), "toxicity"), train_model(
            Feasibility(), "feasibility"
        )

    def analyze(self, smiles: Sequence[str]) -> Sequence[float]:
        return self.safety_model.predict(self.preprocessor.transform(smiles))


class RandomForestBenchmarks:
    filename = "benchmarks_rf"

    def __init__(self):
        self.preprocessor = ChemicalLanguage("bow")
        self.model = cache(MODEL_ROOT / self.filename, self.train)

    def train(self):
        data = Tox21().to_df().fillna(-1)
        model = RandomForestClassifier()

        Xt, Xv, yt, yv = train_test_split(
            self.preprocessor.transform(data["smiles"]),
            data.drop(columns=["smiles", "mol_id"]),
            test_size=0.2,
            random_state=18,
        )

        model.fit(Xt, yt)

        return model

    def analyze(self, smiles: Sequence[str]) -> Sequence[float]:
        return self.model.predict(self.preprocessor.transform(smiles))
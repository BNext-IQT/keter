from typing import Sequence, List
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToInchiKey
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors
from autosklearn.estimators import AutoSklearnRegressor, AutoSklearnClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from keter.datasets.constructed import Safety, Feasibility
from keter.datasets.raw import Tox21, Bbbp
from keter.actors.vectors import ChemicalLanguage
from keter.stage import Stage, ReadOnlyStage


class Analyzer:
    filename = "analyzer"

    def __init__(self, mode="prod", stage: Stage = ReadOnlyStage()):
        model_file = (stage.MODEL_ROOT / f"{self.filename}_{mode}").with_suffix(".pkz")
        if mode != "doc2vec":
            self.preprocessor = ChemicalLanguage("bow", stage=stage)
        elif mode != "lda":
            self.preprocessor = ChemicalLanguage("lda", stage=stage)
        else:
            self.preprocessor = ChemicalLanguage(stage=stage)
        self.stage = stage
        if mode in ["doc2vec", "prod", "lda"]:
            self.safety, self.feasibility, self.bbbp = stage.cache(
                model_file, self.train
            )
        elif mode == "test":
            self.safety, self.feasibility, self.bbbp = self.train(
                score=True, task_duration=900
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def train(self, score=False, task_duration=4500):
        safety_task_duration = task_duration // 2
        feasibility_task_duration = task_duration // 4
        bbbp_task_duration = task_duration // 4

        def train_model(data, target_label, duration, regressor=True):
            dataframe = data.to_df(stage=self.stage)
            if regressor:
                model = AutoSklearnRegressor(time_left_for_this_task=duration)
            else:
                model = AutoSklearnClassifier(time_left_for_this_task=duration)

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

        return (
            train_model(Safety(), "safety", safety_task_duration),
            train_model(Feasibility(), "feasibility", feasibility_task_duration),
            train_model(Bbbp(), "p_np", bbbp_task_duration, regressor=False),
        )

    def analyze(self, smiles: List[str], only_drugs=True) -> pd.DataFrame:
        features = self.preprocessor.transform(smiles)

        # RDKit molecular properties
        inchikey = []
        weight = []
        logp = []
        hdonors = []
        hacceptors = []
        for example in smiles:
            mol = MolFromSmiles(example)
            if not mol:
                raise ValueError("Malformed molecule passed in to analyze")

            inchikey.append(MolToInchiKey(mol))
            weight.append(ExactMolWt(mol))
            logp.append(MolLogP(mol))
            hdonors.append(NumHDonors(mol))
            hacceptors.append(NumHAcceptors(mol))

        # Scores
        safety = self.safety.predict(features)
        feasibility = self.feasibility.predict(features)
        bbbp = self.bbbp.predict_proba(features)

        dataframe = pd.DataFrame(
            {
                "key": inchikey,
                "smiles": smiles,
                "weight": weight,
                "logp": logp,
                "hdonors": hdonors,
                "hacceptors": hacceptors,
                "safety": safety,
                "feasibility": feasibility,
                "bbbp": bbbp,
            }
        )

        if only_drugs:
            # Lipinsky's rules
            dataframe = dataframe[dataframe.weight < 500]
            dataframe = dataframe[dataframe.hdonors <= 5]
            dataframe = dataframe[dataframe.hacceptors <= 10]
            dataframe = dataframe[dataframe.logp <= 5]

            # Filter too toxic and infeasible compounds
            dataframe = dataframe[dataframe.safety > 0.75]
            dataframe = dataframe[dataframe.feasibility > 0.75]

            dataframe = dataframe.reset_index(drop=True)

        return dataframe


class Benchmarks:
    filename = "benchmarks"

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
        _, y_hats = model.predict_proba(Xv)
        scores = []
        print(y_hats)
        scores = roc_auc_score(yv.to_numpy(), y_hats[:, 1], multi_class="ovo")
        # for i, y_hat in enumerate(y_hats):
        #     scores.append(
        #         roc_auc_score(yv.to_numpy()[:, i], y_hat[:, 1], multi_class="ovo")
        #     )
        print(scores)

        return model

    def analyze(self, smiles: Sequence[str]) -> Sequence[float]:
        return self.model.predict(self.preprocessor.transform(smiles))
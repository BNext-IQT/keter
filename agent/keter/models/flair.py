from typing import Iterable
import lzma
import pickle
import pandas as pd
from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer
from flair.data import Sentence, Corpus, Token
from flair.datasets import SentenceDataset
from selfies import encoder
from keter.stage import NullStage
from keter.datasets.raw import Tox21


TRANSFORMED_ROOT = NullStage().DATA_ROOT / "transformed"


class FlairTox21:
    filename = "flair_tox21"

    def to_corpus(self, cache=False) -> Corpus:
        data_file = (TRANSFORMED_ROOT / self.filename).with_suffix(".pickle.xz")
        if data_file.exists():
            with lzma.open(data_file) as fd:
                return pickle.load(fd)

        dataset = Tox21().to_df()

        def plain_tokenizer(text: str) -> Iterable[Token]:
            res = []
            for tok in text.split():
                res.append(Token(tok))
            return res

        def iterate_dataframe(dataset: pd.DataFrame) -> Iterable[Sentence]:
            for _, row in dataset.iterrows():
                res = encoder(row.smiles)
                if not res:
                    continue
                res = res.replace("]", "] ").replace(".", "DOT ")
                sent = Sentence(res.strip(), use_tokenizer=plain_tokenizer)
                for col, val in row.items():
                    if isinstance(val, float):
                        if val == 1.0:
                            sent.add_label(None, col.replace(" ", "_") + "_P ")
                        if val == 0.0:
                            sent.add_label(None, col.replace(" ", "_") + "_N ")
                yield sent

        train = dataset.sample(frac=0.7, random_state=18)
        dataset = dataset.drop(train.index)
        dev = dataset.sample(frac=0.333334, random_state=18)
        test = dataset.drop(dev.index)

        train = SentenceDataset(list(iterate_dataframe(train)))
        dev = SentenceDataset(list(iterate_dataframe(dev)))
        test = SentenceDataset(list(iterate_dataframe(test)))

        corpus = Corpus(train, dev, test, "Molecules")

        if cache:
            TRANSFORMED_ROOT.mkdir(parents=True, exist_ok=True)
            with lzma.open(data_file, "wb") as fd:
                pickle.dump(corpus, fd)

        return corpus


class TARSClassifierModel:
    def fit(self, corpus: Corpus, model_path: str):
        self.model = TARSClassifier(
            task_name="ChemicalUnderstanding",
            label_dictionary=corpus.make_label_dictionary(),
        )

        trainer = ModelTrainer(self.model, corpus)

        trainer.train(
            base_path=model_path,
            learning_rate=0.02,
            mini_batch_size=16,
            mini_batch_chunk_size=4,
            max_epochs=10,
        )

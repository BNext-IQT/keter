from typing import Iterable
import lzma
import pickle
import pandas as pd
from selfies import encoder
from flair.data import Sentence, Corpus, Token
from flair.datasets import SentenceDataset


def save_corpus(filepath: str, corpus: Corpus):
    with lzma.open(filepath, "wb") as corpus_fd:
        pickle.dump(corpus, corpus_fd)


def read_corpus(filepath: str) -> Corpus:
    with lzma.open(filepath) as corpus_fd:
        return pickle.load(corpus_fd)


def transform_elemental(dataset: pd.DataFrame) -> Corpus:
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

    return Corpus(train, dev, test, "Molecules")


from uuid import uuid4
from typing import Sequence, Generator
from multiprocessing import cpu_count
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.matutils import corpus2dense
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keter.operations import generate_smiles2lang


class _WrapGenerator:
    """
    Creates iterators out of replayable generators. Needed for gensim.
    """

    def __init__(self, func):
        self.func = func
        self.generator = func()

    def __iter__(self):
        self.generator = self.func()
        return self

    def __next__(self):
        res = next(self.generator)
        if not res:
            raise StopIteration
        else:
            return res


class ChemicalLanguageHyperparameters:
    """
    Hyperparameters for all chemistry models.
    """

    # Language hyperparams
    max_ngram = 2
    vector_algo = "doc2vec"

    # Doc2Vec hyperparmas
    vec_dims = 460
    vec_window = 4
    max_vocab = 35000
    doc_epochs = 110
    alpha = 0.05

    # LDA hyperparams
    topics = 16

    @staticmethod
    def from_dict(values):
        hp = ChemicalLanguageHyperparameters()
        hp.__dict__.update(values)
        return hp


class ChemicalLanguageModule:
    """
    A chemical language model that creates semantic latent vectors from chemicals,
    based on the mutual information between subtokens of a chemical discriptor and
    a surrigate prediction set encoding chemistry semantics.
    """

    def __init__(self, hyperparams=ChemicalLanguageHyperparameters()):
        self.hyperparams = hyperparams
        if hyperparams.vector_algo not in set(["doc2vec", "lda", "bow"]):
            raise RuntimeError(f"Unsupported algorithm: {hyperparams.vector_algo}")
        self.hyperparams = hyperparams

    def _smiles_to_advanced_lang(
        self, smiles_seq: Generator[str, None, None], training: bool = False
    ) -> Generator[str, None, None]:
        for i, sent in enumerate(smiles_seq):
            sent: Sequence[str] = self._analyzer(sent)
            res: Sequence[str] = []
            for token in sent:
                if token in self.vocab:
                    res.append(token.replace(" ", "A"))
            if training:
                yield TaggedDocument(words=res, tags=[i])
            else:
                yield res

    def _make_iterator(
        self, smiles_seq: Sequence[str], training: bool = False
    ) -> _WrapGenerator:
        return _WrapGenerator(
            lambda: self._smiles_to_advanced_lang(
                generate_smiles2lang(smiles_seq), training
            )
        )

    def make_generator(self, X):
        return self._smiles_to_advanced_lang(generate_smiles2lang(X))

    def to_vecs(self, X: Sequence[str]) -> np.ndarray:
        if self.hyperparams.vector_algo == "lda":
            bows = [self.dictionary.doc2bow(i) for i in self.make_generator(X)]
            latent_vecs, _ = self.topic_model.inference(bows)
            return latent_vecs
        elif self.hyperparams.vector_algo == "doc2vec":
            # Preallocate memory for performance
            latent_vecs = np.empty((len(X), self.document_model.vector_size))

            for i, sent in enumerate(self.make_generator(X)):
                latent_vecs[i] = self.document_model.infer_vector(sent)

            return latent_vecs
        elif self.hyperparams.vector_algo == "bow":
            bows = [self.dictionary.doc2bow(i) for i in self.make_generator(X)]
            return corpus2dense(bows, len(self.dictionary), len(X)).transpose()

    def _fit_language(
        self, X_unmapped: Sequence[str], X: Sequence[str], Y: pd.DataFrame
    ):
        max_featues = min(self.hyperparams.max_vocab, 100000)
        cv = CountVectorizer(
            max_df=0.95,
            min_df=2,
            lowercase=False,
            ngram_range=(1, self.hyperparams.max_ngram),
            max_features=max_featues,
            token_pattern="[a-zA-Z0-9$&+,:;=?@_/~#\\[\\]|<>.^*()%!-]+",
        )

        X_vec = cv.fit_transform(generate_smiles2lang(X))

        local_vocab = set()
        for feat in Y.columns:
            res = zip(
                cv.get_feature_names(),
                mutual_info_classif(X_vec, Y[feat], discrete_features=True),
            )
            local_vocab.update(res)
        self.vocab = {
            i[0]
            for i in sorted(local_vocab, key=lambda i: i[1], reverse=True)[
                : self.hyperparams.max_vocab
            ]
        }

        self._analyzer = cv.build_analyzer()

    def _fit_document_model(
        self, X_unmapped: Sequence[str], X: Sequence[str], Y: pd.DataFrame
    ):
        generator = list(self._make_iterator(X_unmapped, training=True))

        document_model = Doc2Vec(
            vector_size=self.hyperparams.vec_dims,
            alpha=self.hyperparams.alpha,
            workers=max(1, cpu_count() - 2),
            window=self.hyperparams.vec_window,
        )
        document_model.build_vocab(generator)
        document_model.train(
            generator,
            total_examples=len(X_unmapped),
            epochs=self.hyperparams.doc_epochs,
        )

        self.document_model = document_model

    def _fit_topic_model(
        self, X_unmapped: Sequence[str], X: Sequence[str], Y: pd.DataFrame
    ):
        from gensim.models.ldamulticore import LdaMulticore
        from gensim.corpora.dictionary import Dictionary

        iterator = list(self.make_generator(X_unmapped))
        bow = Dictionary(iterator)

        docs = [bow.doc2bow(i) for i in iterator]

        if self.hyperparams.vector_algo == "lda":
            self.topic_model = LdaMulticore(
                docs, id2word=bow, num_topics=self.hyperparams.topics, random_state=18
            )
        self.dictionary = bow

    def fit(self, X_unmapped: Sequence[str], X: Sequence[str], Y: pd.DataFrame):
        self._fit_language(X_unmapped, X, Y)
        if (
            self.hyperparams.vector_algo == "lda"
            or self.hyperparams.vector_algo == "bow"
        ):
            self._fit_topic_model(X_unmapped, X, Y)
        elif self.hyperparams.vector_algo == "doc2vec":
            self._fit_document_model(X_unmapped, X, Y)

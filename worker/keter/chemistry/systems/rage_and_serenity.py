from flair.models.text_classification_model import TARSClassifier
from flair.trainers import ModelTrainer
from flair.data import Corpus


class Serenity:
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


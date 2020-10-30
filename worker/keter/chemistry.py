from typing import Generator
from uuid import uuid4
import pandas as pd
import deepchem as dc
import selfies 

def df2corpus(df: pd.DataFrame) -> Generator:
    if 'smiles' in df:
        df.dropna(subset=['smiles'], inplace=True)
        df['smiles'] = df['smiles'].apply(selfies.encoder)
    

class Chemistry:
    gcm_model_version = '1'

    def __init__(self, path):
        self.molnet_dir = path / 'data' / 'molnet'
        self.gcm_model_dir = path / 'models' / 'chemistry' / 'gcm' / self.gcm_model_version / str(uuid4())
    
    def gather_data(self):
        self.tasks, self.datasets, self.transformers = dc.molnet.load_tox21(featurizer='GraphConv', save_dir=self.molnet_dir)
        self.train, self.verify, self.test = self.datasets

    def fit(self):
        if not hasattr(self, 'train'):
            self.gather_data()
        model = dc.models.GraphConvModel(len(self.tasks), mode='classification', model_dir=self.gcm_model_dir)
        model.fit(self.train, nb_epoch=50)

        self.gcn = model
    
    def score(self):
        metric = dc.metrics.Metric(dc.metrics.roc_auc_score)
        test_metric = self.gcn.evaluate(self.test, [metric], self.transformers)['roc_auc_score']
        train_metric = self.gcn.evaluate(self.train, [metric], self.transformers)['roc_auc_score']
        return test_metric, train_metric

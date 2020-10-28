from typing import Generator
import pandas as pd
import deepchem as dc
import selfies 

def df2corpus(df: pd.DataFrame) -> Generator:
    if 'smiles' in df:
        df.dropna(subset=['smiles'], inplace=True)
        df['smiles'] = df['smiles'].apply(selfies.encoder)
    

class Chemistry:
    def __init__(self, path):
        tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv', save_dir=path)
        train, verify, test = datasets
        model = dc.models.GraphConvModel(len(tasks), mode='classification')
        model.fit(train, nb_epoch=50)

        self.gcn = model
from typing import Generator
import pandas as pd
import selfies 

def df2corpus(df: pd.DataFrame) -> Generator:
    if 'smiles' in df:
        df.dropna(subset=['smiles'], inplace=True)
        df['smiles'] = df['smiles'].apply(selfies.encoder)
    

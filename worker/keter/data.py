from datetime import timedelta
import pandas as pd
from sqlalchemy.engine import Connectable

def get_smiles_from_chembl(conn: Connectable) -> pd.DataFrame:
    query = 'select canonical_smiles as smiles, ' \
            'standard_inchi as inchi, ' \
            'standard_inchi_key as inchi_key ' \
            'from compound_structures'
    return pd.read_sql(query, conn)
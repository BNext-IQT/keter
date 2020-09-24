from datetime import timedelta
import pandas as pd
from sqlalchemy.engine import Connectable

_DATA_VIEWS = {
    'smiles': {
        'sql': 'select canonical_smiles from compound_structures',
        'cache': timedelta(days=180)
    },
    'disease_reports': {
        'sql': 'select * from disease_reports',
        'cache': False
    }
}

def get_dataframe(conn: Connectable, view: str) -> pd.DataFrame:
    sql = _DATA_VIEWS[view]['sql']
    return pd.read_sql(sql, conn)


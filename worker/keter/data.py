import pandas as pd
from functools import reduce


def gather_mols_with_props() -> pd.DataFrame:
    urls = [
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pcba.csv.gz",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
    ]
    result = reduce(
        lambda left, right: pd.merge(left, right, on="smiles", how="outer", sort=False),
        (pd.read_csv(csv) for csv in urls),
    )
    return result[result.smiles != "FAIL"]

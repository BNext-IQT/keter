import pandas as pd
from functools import reduce


def create_supervised_ground_truth():
    supervised = [
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
    ]
    return reduce(
        lambda left, right: pd.merge(left, right, on="smiles", how="outer", sort=False),
        (pd.read_csv(csv) for csv in supervised),
    )

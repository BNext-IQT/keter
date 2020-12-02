from functools import reduce
import lzma
from typing import Iterable
import pandas as pd


def compress(filepath: str, corpus: Iterable[str]):
    with lzma.open(filepath, "wb") as ele_file:
        ele_file.writelines(corpus)


def get_data(with_unlabelled=False):
    urls = [
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pcba.csv.gz",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
    ]

    if with_unlabelled:
        urls.append(
            lambda: pd.read_csv(
                "https://github.com/molecularsets/moses/raw/master/data/dataset_v1.csv",
                names=["smiles", "partition"],
                header=None,
                usecols=["smiles"],
            )
        )

    def read_csv(csv) -> pd.DataFrame:
        if isinstance(csv, str):
            return pd.read_csv(csv)
        return csv()

    result = reduce(
        lambda left, right: pd.merge(left, right, on="smiles", how="outer", sort=False),
        (read_csv(csv) for csv in urls),
    )
    return result[~result.smiles.str.contains(r"^\*|FAIL")].reset_index(drop=True)

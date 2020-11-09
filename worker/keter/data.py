from functools import reduce
import gzip
import pandas as pd
from selfies import encoder


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

    def read_csv(csv) -> pd.DataFrame:
        if isinstance(csv, str):
            return pd.read_csv(csv)
        else:
            return csv()

    result = reduce(
        lambda left, right: pd.merge(left, right, on="smiles", how="outer", sort=False),
        (read_csv(csv) for csv in urls),
    )
    return result[~result.smiles.str.contains("^\*|FAIL")]


def transform_elemental_language(dataset: pd.DataFrame, path: str):
    def transform(dataset: pd.DataFrame) -> str:
        for _, row in dataset.iterrows():
            res = encoder(row.smiles)
            if not res:
                continue
            res = res.replace("[", " ").replace("]", "").replace("\n ", "\n").strip()
            for col, val in row.items():
                if isinstance(val, float):
                    if val == 1.0:
                        res += str(col + "_P ")
                    if val == 0.0:
                        res += str(col + "_N ")
            yield bytes(res, "utf-8")

    with gzip.open(path, "wb") as ele_file:
        ele_file.writelines(transform(dataset))

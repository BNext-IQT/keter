from functools import reduce
import pandas as pd
from keter.cache import cache, CACHE_ROOT


class Molecules:
    DATAFRAME_PATH = CACHE_ROOT / "data" / "molecules.parquet"

    def __call__(self) -> pd.DataFrame:
        if self.DATAFRAME_PATH.exists():
            return pd.read_parquet(self.DATAFRAME_PATH)
        else:
            data = self.download()
            data.to_parquet(self.DATAFRAME_PATH)
            cache([self.DATAFRAME_PATH], "Created new ground truth dataset.")
            return data

    def download(self, with_unlabelled=True) -> pd.DataFrame:
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
            lambda left, right: pd.merge(
                left, right, on="smiles", how="outer", sort=False
            ),
            (read_csv(csv) for csv in urls),
        )
        return result[~result.smiles.str.contains(r"^\*|FAIL")].reset_index(drop=True)

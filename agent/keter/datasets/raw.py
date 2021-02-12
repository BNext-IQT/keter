from functools import reduce
import pandas as pd
from keter.cache import cache, CACHE_ROOT

RAW_DATA_PATH = CACHE_ROOT / "data" / "raw"
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)


class RawData:
    def __call__(self, override=False) -> pd.DataFrame:
        parquet_file = (RAW_DATA_PATH / self.filename).with_suffix(".parquet")
        if parquet_file.exists() and not override:
            dataframe = pd.read_parquet(parquet_file)
        else:
            self.download()
            dataframe.to_parquet(parquet_file)
        return dataframe

    def download(self):
        # All raw data uses CSV at this time
        if ".csv" in self.url:
            self.dataframe = pd.read_csv(self.url)
        else:
            raise EnvironmentError("Only CSV is supported for raw data.")


class Tox21(RawData):
    filename = "tox21"
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"


class ToxCast(RawData):
    filename = "toxcast"
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz"


class ClinTox(RawData):
    filename = "clintox"
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz"


class Sider(RawData):
    filename = "sider"
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"


class Bbbp(RawData):
    filename = "bbbp"
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"


class Pcba(RawData):
    filename = "pcba"
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pcba.csv.gz"


class Muv(RawData):
    filename = "muv"
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz"


class Moses(RawData):
    filename = "moses"
    url = "https://github.com/molecularsets/moses/raw/master/data/dataset_v1.csv"
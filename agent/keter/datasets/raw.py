from functools import reduce
import pandas as pd
from keter.stage import ReadOnlyStage, Stage


class RawData:
    def to_df(self, stage: Stage = ReadOnlyStage()) -> pd.DataFrame:
        path = (stage.DATA_ROOT / "raw" / self.filename).with_suffix(".parquet")
        return stage.cache(path, self.download)

    def download(self) -> pd.DataFrame:
        # All raw data uses CSV at this time
        if ".csv" in self.url:
            dataframe = pd.read_csv(self.url)
        else:
            raise EnvironmentError("Only CSV is supported for raw data.")
        return dataframe


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


class Lipophilicity(RawData):
    filename = "lipophilicity"
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"


class HIV(RawData):
    filename = "hiv"
    url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv"


class ESOL(RawData):
    filename = "esol"
    url = (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
    )


class Moses(RawData):
    filename = "moses"
    url = "https://github.com/molecularsets/moses/raw/master/data/dataset_v1.csv"


class CoronaDeathsUSA(RawData):
    filename = "time_series_covid19_deaths_US"
    url = (
        "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data"
        "/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
    )

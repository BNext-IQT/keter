from pathlib import Path
import pandas as pd
from flask import Flask, render_template
from keter.stage import ReadOnlyStage


def make_drug_db(path: Path) -> pd.DataFrame:
    dataframe = pd.concat(
        [pd.read_parquet(parquet) for parquet in path.glob("*.parquet")],
        axis=0,
        ignore_index=True,
    )
    return dataframe.set_index("key", drop=True)


app = Flask(__name__)
dataframe = make_drug_db(ReadOnlyStage().OUTPUTS_ROOT)


@app.route("/drug/<key>")
def drug(key: str):
    try:
        drug = dataframe.loc[key]
    except:
        abort(404)
    return render_template("chemical.jinja2", drug={"key": key, **drug})

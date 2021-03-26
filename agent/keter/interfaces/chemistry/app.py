from pathlib import Path
import pandas as pd
from flask import Flask, render_template, abort
from flask_frozen import Freezer
from keter.stage import ReadOnlyStage


def make_drug_db(path: Path) -> pd.DataFrame:
    dataframe = pd.concat(
        [pd.read_parquet(parquet) for parquet in path.glob("*.parquet")],
        axis=0,
        ignore_index=True,
    )
    dataframe = dataframe.set_index("key", drop=True)
    dataframe = (
        dataframe.assign(mean=dataframe[["safety", "feasibility"]].mean(axis=1))
        .sort_values("mean", ascending=False)
        .drop("mean", axis=1)
    )
    return dataframe


app = Flask(__name__)
dataframe = make_drug_db(ReadOnlyStage().OUTPUTS_ROOT)


@app.route("/drug/<key>")
def drug(key: str):
    try:
        drug = dataframe.loc[key]
    except:
        abort(404)
    return render_template("chemical.jinja2", drug={"key": key, **drug})


@app.route("/")
def index():
    return render_template("list.jinja2", drugs=dataframe.to_dict("index"))


if __name__ == "__main__":
    app.config["FREEZER_IGNORE_MIMETYPE_WARNINGS"] = True
    app.config["FREEZER_DESTINATION"] = ReadOnlyStage().OUTPUTS_ROOT / "static_html"
    freezer = Freezer(app)
    freezer.freeze()
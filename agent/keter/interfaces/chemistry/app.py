from pathlib import Path
import pandas as pd
from flask import Flask, render_template, abort
from flask_frozen import Freezer
from keter.stage import get_path


class DrugDatabase:
    def lookup(self, key):
        return self.dataframe.loc[key]

    def make_drug_db(self, path: Path):
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
        self.dataframe = dataframe


app = Flask(__name__)
db = DrugDatabase()


@app.before_first_request
def make_drug_db():
    db.make_drug_db(get_path("output"))


@app.route("/drug/<key>")
def drug(key: str):
    try:
        drug = db.lookup(key)
    except:
        abort(404)
    return render_template("chemical.jinja2", drug={"key": key, **drug})


@app.route("/")
def index():
    return render_template("list.jinja2", drugs=db.dataframe.to_dict("index"))


def create_jamstack():
    app.config["FREEZER_IGNORE_MIMETYPE_WARNINGS"] = True
    app.config["FREEZER_DESTINATION"] = get_path("output") / "static_html"
    freezer = Freezer(app)
    freezer.freeze()


if __name__ == "__main__":
    create_jamstack()
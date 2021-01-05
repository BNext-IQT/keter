from flask import Flask, render_template
from keter.schema import DrugCandidate

app = Flask(__name__)


def look_up_drug(name: str) -> DrugCandidate:
    drug = DrugCandidate()
    drug.name = "BNEXT1337"
    drug.key = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    drug.smiles = "CC(=O)Oc1ccccc1C(=O)O"

    drug.drug_score = 18.5
    drug.safety_score = 0.98
    drug.manufacturability_score = 0.99

    drug.mass = 180.16
    drug.logp = 1.31
    drug.hydrogen_acceptors = 1
    drug.hydrogen_donors = 3

    return drug


@app.route("/drug/<name>")
def drug(name: str):
    drug = look_up_drug(name)
    return render_template("chemical.jinja2", drug=drug)


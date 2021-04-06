from tqdm import tqdm
import pandas as pd


def drug_discovery_on_moses(mode="prod"):
    from keter.stage import FileSystemStage
    from keter.actors.sklearn import Analyzer
    from keter.datasets.raw import Moses
    from keter.interfaces.chemistry import create_jamstack

    stage = FileSystemStage()
    if mode == "prod":
        analyzer = Analyzer(stage=stage)
    elif mode == "doc2vec":
        analyzer = Analyzer(mode="doc2vec", stage=stage)
    elif mode == "lda":
        analyzer = Analyzer(mode="lda", stage=stage)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    moses = Moses().to_df(stage=stage)["SMILES"].tolist()

    last = 0
    block_size = 107609
    blocks = []

    stage.OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)

    for i in tqdm(
        range(0, len(moses), block_size), total=len(moses) // block_size, unit="block"
    ):
        blocks.append(analyzer.analyze(moses[i : i + block_size]))
    pd.concat(blocks).reset_index(drop=True).to_parquet(
        stage.OUTPUTS_ROOT / "moses_drugs.parquet"
    )

    create_jamstack()


def drug_discovery_on_moses_bow():
    drug_discovery_on_moses("prod")


def drug_discovery_on_moses_doc2vec():
    drug_discovery_on_moses("doc2vec")


def drug_discovery_on_moses_lda():
    drug_discovery_on_moses("lda")
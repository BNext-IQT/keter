from tqdm import tqdm
import pandas as pd


def drug_discovery_on_moses():
    from keter.stage import FileSystemStage
    from keter.actors.sklearn import Analyzer
    from keter.datasets.raw import Moses
    from keter.interfaces.chemistry import create_jamstack

    stage = FileSystemStage()
    analyzer = Analyzer(mode="prod", stage=stage)
    moses = Moses().to_df(stage=stage)["SMILES"].tolist()

    last = 0
    block_size = 107609
    blocks = []
    for i in tqdm(
        range(0, len(moses), block_size), total=len(moses) // block_size, unit="block"
    ):
        blocks.append(analyzer.analyze(moses[i : i + block_size]))
    pd.concat(blocks).reset_index(drop=True).to_parquet(
        stage.OUTPUTS_ROOT / "moses_drugs.parquet"
    )
    create_jamstack()

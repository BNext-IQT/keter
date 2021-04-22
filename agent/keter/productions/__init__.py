def train_tars(mode="prod"):
    # Interdict the Transformers cache
    # TODO: Make this cleaner somehow
    import os
    from keter.stage import FileSystemStage, get_path

    with FileSystemStage():
        os.environ["TRANSFORMERS_CACHE"] = str(get_path("external") / "transformers")

        from keter.actors.flair import ChemicalUnderstandingTARS

        ChemicalUnderstandingTARS()


def drug_discovery_on_moses(mode="prod"):
    from tqdm import tqdm
    import pandas as pd

    from keter.stage import FileSystemStage, get_path
    from keter.actors.sklearn import Analyzer
    from keter.datasets.raw import Moses
    from keter.interfaces.chemistry import create_jamstack

    with FileSystemStage():
        if mode == "prod":
            analyzer = Analyzer()
        elif mode == "doc2vec":
            analyzer = Analyzer("doc2vec")
        elif mode == "lda":
            analyzer = Analyzer("lda")
        else:
            raise ValueError(f"Invalid mode: {mode}")
        moses = Moses().to_df()["SMILES"].tolist()

        last = 0
        block_size = 107609
        blocks = []

        get_path("output").mkdir(parents=True, exist_ok=True)

        for i in tqdm(
            range(0, len(moses), block_size),
            total=len(moses) // block_size,
            unit="block",
        ):
            blocks.append(analyzer.analyze(moses[i : i + block_size]))
        pd.concat(blocks).reset_index(drop=True).to_parquet(
            get_path("output") / "moses_drugs.parquet"
        )

        create_jamstack()


def drug_discovery_on_moses_bow():
    drug_discovery_on_moses("prod")


def drug_discovery_on_moses_doc2vec():
    drug_discovery_on_moses("doc2vec")


def drug_discovery_on_moses_lda():
    drug_discovery_on_moses("lda")
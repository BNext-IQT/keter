import cython
import pd
from typing import Sequence, Generator
from selfies import encoder


def transform_low_token_language(dataset: pd.DataFrame) -> str:
    def transform_row(row: pd.Series) -> str:
        return encoder(row.smiles).replace("[", " ").replace("]", "").lstrip() + [
            i for i in row
        ]

    return dataset.apply(
        lambda row: encoder(row.smiles).replace("[", " ").replace("]", "").lstrip(),
        axis=1,
    ).to_string(index=False)


def smiles2lang(smiles: str) -> str:
    """
    High performance SMILES parser that can be compiled into machine code for further speedup.
    """
    res: list = []
    token: str = ""
    inbracket: cython.bint = False
    maybe_chlorine: cython.bint = False
    maybe_bromine: cython.bint = False
    for char in smiles:
        if maybe_chlorine:
            maybe_chlorine = False
            if char == "l":
                res.append("Cl")
                continue
            else:
                res.append("C")
        if maybe_bromine:
            maybe_bromine = False
            if char == "r":
                res.append("Br")
                continue
            else:
                res.append("B")
        if char == "]":
            res.append(token + "]")
            token = ""
            inbracket = False
        elif char == "[" or inbracket:
            token += char
            inbracket = True
        elif char == "C":
            maybe_chlorine = True
        elif char == "B":
            maybe_bromine = True
        else:
            res.append(char)
    if maybe_chlorine:
        res.append("C")
    if maybe_bromine:
        res.append("B")
    return " ".join(res)


def trivial_generator(smiles_seq: Sequence[str]) -> Generator[str, None, None]:
    for smiles in smiles_seq:
        yield smiles2lang(smiles)

from libc.stdlib cimport rand
from typing import Sequence, Generator
import numpy as np
import cython


def construct_infection_records(
    date: cython.int, deaths: cython.int, lat: cython.float, long_: cython.float
) -> list:
    res: list = []
    # 86 = IFR of about 1.15%
    infections = deaths * 86
    for i in range(infections):
        # 216000 = 2.5 days in seconds
        # 1296000 = 15 days in seconds
        case_timestamp: cython.int = date + rand() % 216001 - 1296000
        case_lat: cython.float = lat + (rand() % 2000 - 1000) / 3000
        case_long: cython.float = long_ + (rand() % 2000 - 1000) / 3000
        res.append(f"{case_timestamp}, {case_lat}, {case_long}")
    return res


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


def generate_smiles2lang(smiles_seq: Sequence[str]) -> Generator[str, None, None]:
    for smiles in smiles_seq:
        yield smiles2lang(smiles)
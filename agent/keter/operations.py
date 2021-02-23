import numpy as np
import cython


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
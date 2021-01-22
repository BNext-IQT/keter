from typing import Sequence
from rdkit import Chem


def smiles2key(smiles: str) -> str:
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))


def smiles2keys(smiles: Sequence) -> Sequence:
    mols = Chem.SmilesMolSupplierFromText("\n".join(smiles), " ", 0, -1, 0)
    for mol in mols:
        yield Chem.MolToInchiKey(mol)

from rdkit import Chem


def smiles2key(smiles: str) -> str:
    return Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))
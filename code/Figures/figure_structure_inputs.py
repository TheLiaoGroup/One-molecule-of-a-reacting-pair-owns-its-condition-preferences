"""Exact reaction structures, checked against records and a single C-X/N-H edit."""
from pathlib import Path

import pandas as pd
from rdkit import Chem


SPECS = {
    1: ("Buchwald-Hartwig|1", "N#Cc1ccnc(Cl)c1", "Cc1cc(C)nc(N)c1", "Cl"),
    2: ("SNAr|1", "Cc1cc(F)ccc1C#N", "c1cc2[nH]ccc2cn1", "F"),
}
COLUMNS = ["role", "smiles", "source_screen", "source_pair"]


def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None, f"Invalid structure: {smiles}"
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def substitution(halide, nucleophile, leaving):
    a, b = Chem.MolFromSmiles(halide), Chem.MolFromSmiles(nucleophile)
    assert a is not None and b is not None
    xs = [atom for atom in a.GetAtoms() if atom.GetSymbol() == leaving
          and atom.GetDegree() == 1 and atom.GetNeighbors()[0].GetIsAromatic()]
    ns = [atom for atom in b.GetAtoms() if atom.GetSymbol() == "N" and atom.GetTotalNumHs() > 0]
    assert len(xs) == len(ns) == 1, "The depicted reaction must have one unambiguous C-X/N-H edit"
    x, n = xs[0], ns[0]
    joined = Chem.RWMol(Chem.CombineMols(a, b))
    target_n = joined.GetAtomWithIdx(a.GetNumAtoms() + n.GetIdx())
    if target_n.GetNumExplicitHs():
        target_n.SetNumExplicitHs(target_n.GetNumExplicitHs() - 1)
    joined.AddBond(x.GetNeighbors()[0].GetIdx(), a.GetNumAtoms() + n.GetIdx(), Chem.BondType.SINGLE)
    joined.RemoveAtom(x.GetIdx())
    molecule = joined.GetMol()
    Chem.SanitizeMol(molecule)
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def selected_pair(records, number):
    screen, halide, nucleophile, _ = SPECS[number]
    pair = records[records.screen.eq(screen)
                   & records.sub_1_smiles.eq(nucleophile)
                   & records.sub_2_smiles.eq(halide)].copy()
    assert len(pair) > 0 and pair.subpair.nunique() == 1, f"Figure {number}: depicted pair absent"
    return pair


def from_raw(records, number):
    screen, halide, nucleophile, leaving = SPECS[number]
    pair = selected_pair(records, number)
    products = pair.product_smiles.dropna().unique()
    assert len(products) == 1, f"Figure {number}: inconsistent recorded product"
    assert canonical(products[0]) == substitution(halide, nucleophile, leaving), "Recorded product connectivity differs"
    return pd.DataFrame([
        dict(role=role, smiles=smiles, source_screen=screen, source_pair=pair.subpair.iloc[0])
        for role, smiles in (("halide", halide), ("N-H partner", nucleophile), ("product", products[0]))
    ], columns=COLUMNS)


def validate(table, records, number):
    assert list(table.columns) == COLUMNS and list(table.role) == ["halide", "N-H partner", "product"]
    screen, halide, nucleophile, leaving = SPECS[number]
    values = table.set_index("role").smiles.to_dict()
    assert canonical(values["halide"]) == canonical(halide)
    assert canonical(values["N-H partner"]) == canonical(nucleophile)
    pair = selected_pair(records, number)
    assert table.source_screen.eq(screen).all() and table.source_pair.eq(pair.subpair.iloc[0]).all()
    assert canonical(values["product"]) == substitution(pair.sub_2_smiles.iloc[0], pair.sub_1_smiles.iloc[0], leaving), "Structural input product differs"
    return values


def load(root, number, records=None):
    root = Path(root)
    if records is None:
        records = pd.read_csv(root / "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz",
                              usecols=["screen", "subpair", "sub_1_smiles", "sub_2_smiles"])
    table = pd.read_csv(root / f"07_METHODS_AND_PROVENANCE/figure_inputs/figure{number}_structures.csv")
    return validate(table, records, number)


def main():
    root = Path(__file__).resolve().parents[2]
    records = pd.read_csv(root / "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz")
    folder = root / "07_METHODS_AND_PROVENANCE/figure_inputs"
    folder.mkdir(exist_ok=True)
    for number in SPECS:
        table = from_raw(records, number)
        table.to_csv(folder / f"figure{number}_structures.csv", index=False, lineterminator="\n")
        validate(table, records, number)
        print(f"Figure {number}: recorded product equals independently edited reactants")


if __name__ == "__main__":
    main()

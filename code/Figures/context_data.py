"""Retained scientific extraction, independent of figure rendering."""
import re
import hashlib
import numpy as np
import pandas as pd
from rdkit import Chem
import figure_common as c

R = c.ROOT


D_FILE = "05_DATA/Results/directional/phase1_D.csv"


T_FILE = "05_DATA/Results/directional/phase1_T.csv"


S_FILE = "05_DATA/Results/directional/phase1_summary.json"


COHORTS = "05_DATA/Analysis/directional_cohorts/DESIGN_COUNTS.csv"


MANIFEST = "02_FIGURES/ChemDraw/structures_manifest.csv"


STRUCTURE_SPEC = "07_METHODS_AND_PROVENANCE/FIGURE_STRUCTURES_PROMOTION_READY.csv"


NAME = {
    "CSc1cccc(N)c1": "3-(methylthio)aniline", "COc1ccc(N)cc1OC": "3,4-dimethoxyaniline",
    "CCOc1cc(N)ccn1": "2-ethoxypyridin-4-amine",
    "Cc1cn(-c2cc(N)cc(C(F)(F)F)c2)cn1": "3-(4-methylimidazol-1-yl)-\n5-(trifluoromethyl)aniline",
    "Cc1ccc(N)c(C)c1": "2,4-dimethylaniline", "Cc1ccc(N)nc1": "5-methylpyridin-2-amine",
    "COc1ccc(N)cc1": "4-methoxyaniline", "CNc1ccc(OC)cc1": "4-methoxy-N-methylaniline",
    "C#Cc1cccc(N)c1": "3-ethynylaniline", "CSc1ccc(N)cc1": "4-(methylthio)aniline",
    "Cc1ccc(N)cc1": "4-methylaniline", "COc1ccc(N)cn1": "6-methoxypyridin-3-amine",
    "CCCCN1CCNCC1": "1-butylpiperazine", "CCOC(=O)c1cccc(N)c1": "Ethyl 3-aminobenzoate",
    "COc1ccc(N)cc1C": "4-methoxy-3-methylaniline", "CCNc1ccccc1": "N-ethylaniline",
    "Cc1ccccc1N": "2-methylaniline", "Nc1ccc2nccnc2c1": "Quinoxalin-6-amine",
    "COC(=O)c1ccc(N)cc1OC": "Methyl 4-amino-2-methoxybenzoate",
    "COc1cc(C(=O)O)cc(OC)c1OC": "3,4,5-trimethoxybenzoic acid",
    "O=C(O)c1ccc(C(F)(F)F)cn1": "5-(trifluoromethyl)picolinic acid",
    "COc1ncc(Br)cc1C(=O)O": "5-bromo-2-methoxynicotinic acid",
    "O=C(O)c1ccc(F)cn1": "5-fluoropicolinic acid",
}


def molecule_name(smiles, styled=False):
    name = NAME[smiles].lower()
    name = re.sub(r"(^|[-,(])n(?=[-,])", lambda m: m.group(1) + "N", name)
    if styled:
        name = re.sub(r"(^|[-,(])N(?=[-,])", lambda m: m.group(1) + r"$\mathit{N}$", name)
    return name


def load_data():
    import json
    a = pd.read_csv(R / D_FILE)
    assert len(a) == 23 and set(a.molecule) == set(NAME)
    a = pd.concat([a[a.stratum == key].sort_values(["D", "molecule"], ascending=[False, True]) for key in ("amine", "acid")], ignore_index=True)
    a["plot_row"] = np.arange(1, 24)
    a["name"] = a.molecule.map(molecule_name).str.replace("\n", "", regex=False)
    a["name_display"] = a.molecule.map(lambda m: molecule_name(m, styled=True))
    t = pd.read_csv(R / T_FILE)
    assert len(t) > 0 and not t.duplicated(["molecule", "array", "role"]).any()
    cohorts = pd.read_csv(R / COHORTS)
    primary = cohorts[(cohorts.set == "G1") & cohorts.array_clear_owner].copy()
    contexts = primary.merge(t, on=["molecule", "array", "role"], how="left", validate="one_to_one", suffixes=("_design", ""))
    assert contexts["T"].notna().all() and len(contexts) == 55
    for row in a.itertuples():
        group = contexts[contexts.molecule == row.molecule]
        owner = group.loc[group.context == "owner", "T"].mean()
        nonowner = group.loc[group.context == "non-owner", "T"].mean()
        assert np.allclose([owner, nonowner, owner - nonowner], [row.T_O, row.T_N, row.D], atol=1e-12, rtol=0)
    manifest = pd.read_csv(R / MANIFEST)
    structures = manifest[manifest.name.isin([f"Fig2_b_M{i}" for i in range(1, 9)])].set_index("label")
    spec = pd.read_csv(R / STRUCTURE_SPEC)
    spec = spec[spec.label.isin([f"M{i}" for i in range(1, 9)])].set_index("label")
    assert len(spec) == 8 and spec.index.is_unique
    tri = sorted(cohorts.loc[cohorts.triple_context, "molecule"].unique(), key=lambda m: (Chem.MolFromSmiles(m).GetNumHeavyAtoms(), m))
    assert len(tri) == 8
    lookup = t.set_index(["molecule", "array", "role"])
    partner_counts = cohorts.set_index(["molecule", "array", "role"]).n_pf_partners
    rows = []
    for i, molecule in enumerate(tri, 1):
        label = f"M{i}"
        assert structures.loc[label, "smiles"] == spec.loc[label, "smiles"] == molecule
        for suffix in ("svg", "cdxml"):
            path = R / f"02_FIGURES/ChemDraw/Fig2_b_{label}.{suffix}"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == structures.loc[label, f"{suffix}_sha256"]
        values = {col: float(lookup.loc[(molecule, array, 1), "T"]) for col, array in
                  (("T_amide", "Acylation|1"), ("T_BH1", "Buchwald-Hartwig|1"), ("T_BH2", "Buchwald-Hartwig|2"))}
        counts = {f"partners_{key}": int(partner_counts.loc[(molecule, array, 1)]) for key, array in
                  (("amide", "Acylation|1"), ("BH1", "Buchwald-Hartwig|1"), ("BH2", "Buchwald-Hartwig|2"))}
        cn = (values["T_BH1"] + values["T_BH2"]) / 2
        diff = values["T_amide"] - cn
        rows.append(dict(row_type="amine", plot_label=label, name=molecule_name(molecule), smiles=molecule,
                         heavy_atoms=Chem.MolFromSmiles(molecule).GetNumHeavyAtoms(), **values, CN_mean=cn, R=diff,
                         amide_above_CN_mean=diff > 0, amide_above_both=values["T_amide"] > max(values["T_BH1"], values["T_BH2"]),
                         **counts, line_colour="exception" if label in ("M3", "M8") else "standard", count=np.nan, of=np.nan))
    b = pd.DataFrame(rows)
    summary = json.loads((R / S_FILE).read_text(encoding="utf-8"))
    assert abs(a.D.mean() - summary["A"]["mean"]) < 1e-12
    assert abs(b.R.mean() - summary["C"]["mean_R"]) < 1e-12
    assert set(b.loc[b.R < 0, "plot_label"]) == {"M3", "M8"}
    assert abs((b.T_amide - b.T_BH1).mean() - summary["C"]["mean_amide_minus_BH1"]) < 1e-12
    assert abs((b.T_amide - b.T_BH2).mean() - summary["C"]["mean_amide_minus_BH2"]) < 1e-12
    return a, b, t, summary, contexts

"""Retained scientific extraction, independent of figure rendering."""
import numpy as np
import pandas as pd
from rdkit import Chem
import figure_common as c

R = c.ROOT


RESULT = "05_DATA/Results/display_inputs/Anatomy_arrays.csv"


RAW = "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz"


CONDITIONS = "10_PUBLIC_RELEASE_CANDIDATE/data/conditions.csv"


ROLES = "05_DATA/Results/primary/history_roles.csv"


ARRAY_ORDER = ["Buchwald-Hartwig|1", "Buchwald-Hartwig|2", "C-O competitive reactions|1", "Heck|1", "Hiyama|1",
               "Kumada|1", "SNAr|1", "Sonogashira|1", "Suzuki|1", "Ullmann|1", "Acylation|1", "Chan-Lam|1",
               "Esterification|1", "Grignard|1", "Reductive-amination|1"]


HALIDE = "Cc1cc(F)ccc1C#N"


NUC = "c1cc2[nH]ccc2cn1"


def role_key(s):
    if "halide" in s:
        return "halide"
    if s in ("N-H partner", "N-H nucleophile", "amine"):
        return "amine"
    if s == "carboxylic acid":
        return "acid"
    return "other"


def plain(s):
    return str(s).replace("\N{EN DASH}", "-")


def chemical_text(s):
    for old, new in (("C-N", "C\u2013N"), ("N-H", "N\u2013H"), ("C-O", "C\u2013O"), ("Chan-Lam", "Chan\u2013Lam")):
        s = str(s).replace(old, new)
    return s


SHORT = {"N-H partner": "N-H", "N-H nucleophile": "N-H", "aryl (pseudo)halide": "halide",
         "aryl halide": "halide", "aryl or acyl halide": "halide", "phenol or alcohol": "alcohol/phenol",
         "carboxylic acid": "acid", "Grignard reagent": "Grignard", "carbonyl compound": "carbonyl",
         "boronic acid": "boronic acid", "organosilane": "silane"}


ADDITIVE_GROUPS = [
    ("Acids", ["AcOH", "MsOH", "PTSA", "TFA"]),
    ("No\nadditive", ["none"]),
    ("Amine bases", ["DIPEA", "TEA", "2,6-lutidine", "Py", "DBU"]),
    ("Carbonates, bicarbonate\nand acetate", ["Cs2CO3", "K2CO3", "KHCO3", "KOAc"]),
    ("Hydroxide and\nalkoxides", ["KOH", "KOtBu", "NaOMe", "NaOtAm"]),
]


ADDITIVE_LABELS = {"none": "none", "Cs2CO3": "Cs$_2$CO$_3$", "K2CO3": "K$_2$CO$_3$", "KHCO3": "KHCO$_3$"}


SOLVENTS = ["DMSO", "THF", "t-AmOH", "NMP", "CPME", "MeCN"]


def load_data():
    a = pd.read_csv(R / RESULT).set_index("screen").loc[ARRAY_ORDER].reset_index()
    roles = pd.read_csv(R / ROLES).set_index("screen")
    assert len(a) == len(roles) == 15 and set(a.screen) == set(roles.index)
    a["plot_row"] = np.arange(1, 16)
    a["array"] = a.screen.map(lambda key: c.array_label(key, mathtext=False))
    a["array_display"] = a.screen.map(c.array_label)
    a["role1"] = roles.loc[a.screen, "role_1"].map(plain).to_numpy()
    a["role2"] = roles.loc[a.screen, "role_2"].map(plain).to_numpy()
    a["halide_array"] = a.electrophile_role.notna()
    raw = pd.read_csv(R / RAW, usecols=["screen", "condition_id", "subpair", "sub_1_smiles", "sub_2_smiles", "yield"])
    from figure_structure_inputs import load
    structures = load(R, 2, raw)
    leaving = [(label, Chem.MolFromSmarts(smarts)) for label, smarts in
               (("I", "cI"), ("Br", "cBr"), ("Cl", "cCl"), ("F", "cF"), ("sulfonate", "cOS(=O)(=O)"))]
    flags = []
    for row in a.itertuples():
        if not row.halide_array:
            flags.append(False)
            continue
        col = f"sub_{int(row.electrophile_role)}_smiles"
        mols = [Chem.MolFromSmiles(s) for s in raw.loc[raw.screen == row.screen, col].unique()]
        assert len(mols) > 0 and all(m is not None for m in mols)
        groups = [next((label for label, pattern in leaving if m.HasSubstructMatch(pattern)), "other") for m in mols]
        flags.append(set(groups) == {"Br"})
    a["aryl_bromide_only"] = flags
    assert set(a.loc[a.aryl_bromide_only, "screen"]) == {"Buchwald-Hartwig|2", "Heck|1", "Sonogashira|1", "Suzuki|1"}
    pair = raw[(raw.screen == "SNAr|1") & (raw.sub_1_smiles == NUC) & (raw.sub_2_smiles == HALIDE)].copy()
    assert pair.condition_id.nunique() == len(pair) == 47
    pair["product_smiles"] = structures["product"]
    pair = pair[["screen", "condition_id", "sub_1_smiles", "sub_2_smiles", "product_smiles", "yield"]]
    cond = pd.read_csv(R / CONDITIONS)
    cond = cond[cond.rxn_class == "SNAr"].copy()
    assert len(cond) == 47 and set(cond.condition_id) == set(pair.condition_id)
    assert all(cond[f"Reagent{i}_id"].isna().all() for i in (1, 2, 4, 5, 6))
    cond["base_or_acid"] = cond["Reagent3_Standardized Name"].fillna("none")
    n_add, n_sol = cond.base_or_acid.nunique(), cond["Standardized Name"].nunique()
    factorial = len(cond) == n_add * n_sol and not cond.duplicated(["base_or_acid", "Standardized Name"]).any()
    assert not factorial
    columns = [name for group, names in ADDITIVE_GROUPS for name in names]
    classes = {name: group.replace("\n", " ") for group, names in ADDITIVE_GROUPS for name in names}
    assert set(columns) == set(cond.base_or_acid) and len(columns) == 18
    assert not cond.duplicated(["base_or_acid", "Standardized Name"]).any(), "Different recipes collide in the proposed matrix"
    cond["matrix_column"] = cond.base_or_acid.map({name: i + 1 for i, name in enumerate(columns)})
    cond["matrix_row"] = cond["Standardized Name"].map({name: i + 1 for i, name in enumerate(SOLVENTS)})
    cond["additive_class"] = cond.base_or_acid.map(classes)
    cond["additive_display_name"] = cond.base_or_acid.map(lambda name: ADDITIVE_LABELS.get(name, name))
    assert cond[["matrix_column", "matrix_row"]].notna().all().all()
    return a, cond, pair

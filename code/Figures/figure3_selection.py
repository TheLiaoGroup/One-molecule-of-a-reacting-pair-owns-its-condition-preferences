"""Frozen Rule 2 display selection; no model or role-share estimates change."""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import figure_common as common
import structure_art as sa

BASE_ART_LOAD = sa.load

RULE = ROOT / "07_METHODS_AND_PROVENANCE/frozen_analysis_plans/figure_selection_v3/RULE.md"
RULE_SHA = "318b4babc7d8918fcc1705d762861ebc40278111ddd82e093b10b3b32f89a721"
RAW = "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz"
RESULT = "05_DATA/Results/display_inputs/Anatomy_arrays.csv"
ROLES = "05_DATA/Results/primary/history_roles.csv"
ARRAY_ORDER = ["Buchwald-Hartwig|1", "Buchwald-Hartwig|2", "C-O competitive reactions|1", "Heck|1", "Hiyama|1",
               "Kumada|1", "SNAr|1", "Sonogashira|1", "Suzuki|1", "Ullmann|1", "Acylation|1", "Chan-Lam|1",
               "Esterification|1", "Grignard|1", "Reductive-amination|1"]
NAMES = {
    "N#Cc1ccnc(Cl)c1": "2-chloro-4-cyanopyridine",
    "O=C(c1ccccc1)c1ccccc1Br": "2-bromobenzophenone",
    "Cc1cc(Br)ccn1": "4-bromo-2-methylpyridine",
    "Brc1ccc2ncnn2c1": "bromotriazolopyridine (exact isomer specified by SMILES)",
    "COc1cccnc1Br": "2-bromo-3-methoxypyridine",
    "COc1cccc(Cl)n1": "2-chloro-6-methoxypyridine",
    "O=[N+]([O-])c1ccccc1F": "1-fluoro-2-nitrobenzene",
    "Brc1ccc2cn[nH]c2c1": "bromo-1H-indazole (exact isomer specified by SMILES)",
    "Brc1cccc2cc[nH]c12": "7-bromo-1H-indole",
    "C#CC(C)(C)N": "2-methylbut-3-yn-2-amine",
    "COc1ccc(OC)c(B(O)O)c1": "2,5-dimethoxyphenylboronic acid",
    "COc1cc(C(=O)O)cc(OC)c1OC": "3,4,5-trimethoxybenzoic acid",
    "O=Cc1ccnc2ccccc12": "quinoline-4-carbaldehyde",
    "NC1CCN(C(=O)OCc2ccccc2)CC1": "benzyl 4-aminopiperidine-1-carboxylate",
}


def load_art(path):
    """Extend the existing parser only for genuine ChemDraw superscript charges."""
    try:
        return BASE_ART_LOAD(path)
    except AssertionError:
        root = ET.parse(path).getroot()
        width, height = (float(root.get(key).removesuffix("px")) for key in ("width", "height"))
        polygons, labels = [], []
        for element in root:
            tag = element.tag.removeprefix(sa.NS)
            if tag == "path":
                assert element.get("fill") == "#000000" and element.get("stroke") == "none"
                assert set(re.findall(r"[A-Za-z]", element.get("d"))) <= set("MLZe")
                polygons.append(sa._polygon(element.get("d"), sa._matrix(element.get("transform"))))
            elif tag == "text":
                assert len(element) == 0 and element.get("font-family") == "Arial"
                assert element.get("font-style") == element.get("font-weight") == "normal"
                matrix = sa._matrix(element.get("transform"))
                assert abs(matrix[0, 0] - matrix[1, 1]) < 1e-9 and matrix[0, 1] == matrix[1, 0] == 0
                size = float(element.get("font-size").removesuffix("px")) * matrix[0, 0]
                labels.append(dict(text=element.text.strip(), x=matrix[0, 2], y=matrix[1, 2], size=size))
            else:
                raise ValueError(f"Unexpected ChemDraw SVG element {tag}")
        main = max(label["size"] for label in labels)
        assert any(label["text"] in ("+", "-", "\u2212") for label in labels)
        for label in labels:
            scale = label["size"] / main
            assert abs(scale - 1) < 1e-6 or (abs(scale - 0.75) < 1e-6 and
                (label["text"].isdigit() or label["text"] in ("+", "-", "\u2212"))), label
            label["scale"] = 1.0 if abs(scale - 1) < 1e-6 else 0.75
        return dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(), width=width,
                    height=height, polygons=polygons, labels=labels, label_px=round(main, 6))


def verify_rule():
    actual = hashlib.sha256(RULE.read_bytes()).hexdigest()
    assert actual == RULE_SHA, (actual, RULE_SHA)
    return actual


def consistency(x, y):
    jointly_finite = np.isfinite(x) & np.isfinite(y)
    if jointly_finite.sum() < 2:
        return float("nan")
    a, b = x[jointly_finite], y[jointly_finite]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def calculate():
    verify_rule()
    shares = pd.read_csv(ROOT / RESULT).set_index("screen").loc[ARRAY_ORDER]
    roles = pd.read_csv(ROOT / ROLES).set_index("screen")
    raw = pd.read_csv(ROOT / RAW, usecols=["screen", "condition_id", "subpair", "sub_1_smiles", "sub_2_smiles", "yield"])
    raw["yield"] = pd.to_numeric(raw["yield"], errors="coerce")
    raw = raw[raw["yield"].notna()].copy()
    assert (raw.subpair == raw.sub_1_smiles + " >> " + raw.sub_2_smiles).all()
    selections, candidates = [], []
    for screen in ARRAY_ORDER:
        role = int(shares.loc[screen, "owner_role"])
        role_name = str(roles.loc[screen, f"role_{role}"])
        group = raw[raw.screen.eq(screen)]
        pivot = group.pivot(index="subpair", columns="condition_id", values="yield")
        pivot = pivot[sorted(pivot.columns, key=lambda key: int(key[1:]))]
        matrix = pivot.to_numpy(float)
        active = np.nanmax(matrix, axis=1) >= 0.10
        matrix = matrix[active]
        signature = np.nanmean(matrix, axis=0)
        residual = np.full_like(matrix, np.nan)
        # Match figure1.measurement_tables normal-equation OLS, including operation order.
        for i in range(len(matrix)):
            observed = np.isfinite(matrix[i]) & np.isfinite(signature)
            design = np.c_[np.ones(observed.sum()), signature[observed]]
            inverse = np.linalg.pinv(design.T @ design)
            coefficients = inverse @ design.T @ matrix[i, observed]
            residual[i, observed] = matrix[i, observed] - design @ coefficients
        pairs = np.asarray(pivot.index[active])
        molecules = np.asarray([pair.split(" >> ")[role - 1] for pair in pairs])
        all_owners = sorted(group[f"sub_{role}_smiles"].unique())
        local = []
        for smiles in all_owners:
            indices = np.where(molecules == smiles)[0]
            correlations = [consistency(100 * residual[i], 100 * residual[j])
                            for i, j in itertools.combinations(indices, 2)]
            mean = float(np.mean(correlations)) if correlations else float("nan")
            local.append(dict(screen=screen, array=common.array_label(screen, mathtext=False), owner_role=role,
                owner_role_name=role_name, smiles=smiles, name=NAMES.get(smiles, smiles),
                name_source="curated structure name" if smiles in NAMES else "original SMILES identifier",
                product_forming_partners=len(indices), pairwise_correlations=correlations,
                mean_pairwise_r=mean))
        threshold = next((k for k in (4, 3, 2) if any(c["product_forming_partners"] >= k for c in local)), None)
        assert threshold is not None, f"Rule 2 stopped: {screen} has no owner with two product-forming partners"
        eligible = [c for c in local if c["product_forming_partners"] >= threshold and np.isfinite(c["mean_pairwise_r"])]
        assert eligible, f"Rule 2 stopped: {screen} has no finite consistency at threshold {threshold}"
        median = float(np.median([c["mean_pairwise_r"] for c in eligible]))
        for c in local:
            c.update(threshold_used=threshold, eligible=c in eligible, role_median_r=median,
                     absolute_distance_to_median=abs(c["mean_pairwise_r"] - median), selected=False)
        chosen = min(eligible, key=lambda c: (c["absolute_distance_to_median"], -c["product_forming_partners"], c["smiles"]))
        chosen["selected"] = True
        selections.append({k: v for k, v in chosen.items() if k != "pairwise_correlations"})
        candidates.extend(local)
    return dict(rule_file=str(RULE.relative_to(ROOT)).replace("\\", "/"), rule_sha256=RULE_SHA,
                definition="Existing frozen owner roles; max yield >= 0.10; normal-equation OLS descriptive residuals; unweighted mean pairwise Pearson r; jointly finite conditions.",
                selection=selections, all_candidates=candidates)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value



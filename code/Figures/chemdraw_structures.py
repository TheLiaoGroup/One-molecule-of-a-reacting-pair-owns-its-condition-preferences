"""Chemical structures for the figures, drawn by ChemDraw's own engine with ACS Document 1996 settings.

Molecules come from 07_METHODS_AND_PROVENANCE/FIGURE_STRUCTURES_PROMOTION_READY.csv. RDKit only fixes the orientation
(the reacting group faces the partner, rings stand upright) and the Kekule form; ChemDraw (ChemDrawControl22, installed with ChemOffice 2022)
lays out labels and bonds and writes CDXML (editable) to 02_FIGURES/ChemDraw/; the SVG placed in the figure is always
exported by ChemDraw from that CDXML. A CDXML edited by hand in ChemDraw is never overwritten (its hash no longer matches
the manifest): the script re-exports its SVG, so a hand edit reaches the figure on the next run.
    python 06_CODE/Figures/chemdraw_structures.py 1
"""
import csv
import hashlib
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import structure_art                                              # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_FIGURES/ChemDraw"
SPEC = ROOT / "07_METHODS_AND_PROVENANCE/FIGURE_STRUCTURES_PROMOTION_READY.csv"
MANIFEST = OUT / "structures_manifest.csv"
RIGHT, LEFT, UP = 0.0, np.pi, np.pi / 2                         # direction the reacting group faces
ANCHOR = {"N-H partner": ("[NX3;H1,H2]", RIGHT), "carboxylic acid": ("[CX3](=O)[OX2H1]", LEFT),
          "N-H nucleophile": ("[#7;!H0]", RIGHT), "aryl halide": ("[F,Cl,Br,I]c", LEFT)}
PANEL_FACE = {("2", "b"): UP}                                      # the eight amines stand in one row under panel 2b, NH2 up
ROLE_FACE = {("1", "a", "aryl halide"): RIGHT, ("1", "a", "N-H partner"): LEFT}   # panel 1a: the halide on the left faces the amine on the right
GENERATED = "ChemDraw 22 (ChemDrawControl22), ACS Document 1996 settings; RDKit orientation"
EDITED = "edited by hand in ChemDraw; SVG re-exported by ChemDraw"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rdDepictor.SetPreferCoordGen(True)


def square_up(m, Q, anchor_atom):
    """The smallest turn that stands the molecule upright: the six-membered ring nearest the reacting group with its edges on
    multiples of 30 degrees (two edges vertical or two horizontal, substituent bonds on the same grid); failing one, the
    five-membered ring nearest it with its symmetry axis horizontal or vertical; failing a ring, the bonds of a chain on
    the 30-degree grid."""
    near = lambda r: np.linalg.norm(Q[list(r)].mean(0) - Q[anchor_atom])
    six = [r for r in m.GetRingInfo().AtomRings() if len(r) == 6]
    five = [r for r in m.GetRingInfo().AtomRings() if len(r) == 5]
    if six:                                                       # a ring edge on the 30-degree grid
        ring = min(six, key=near)
        v, grid = Q[ring[1]] - Q[ring[0]], 30
    elif five:                                                    # the pentagon's symmetry axis horizontal or vertical
        ring = min(five, key=near)
        apex = min(ring, key=lambda a: np.linalg.norm(Q[a] - Q[anchor_atom]))
        v, grid = Q[apex] - Q[list(ring)].mean(0), 90
    else:                                                         # a chain: its bonds on the 30-degree grid
        v, grid = Q[m.GetAtomWithIdx(anchor_atom).GetNeighbors()[0].GetIdx()] - Q[anchor_atom], 30
    phi = np.degrees(np.arctan2(v[1], v[0]))
    d = np.radians(-((phi + grid / 2) % grid - grid / 2))
    return np.array([[np.cos(d), -np.sin(d)], [np.sin(d), np.cos(d)]])


def oriented_molblock(smiles, anchor, angle):
    m = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(m)
    c = m.GetConformer()
    P = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y] for i in range(m.GetNumAtoms())])
    hit = m.GetSubstructMatch(Chem.MolFromSmarts(anchor))
    assert hit, (smiles, anchor)
    v = P[hit[0]] - P.mean(0)
    th = angle - np.arctan2(v[1], v[0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q = (P - P.mean(0)) @ R.T
    Q = Q @ square_up(m, Q, hit[0]).T                            # rings upright: every ring edge at a multiple of 30 degrees
    acid = m.GetSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1]"))
    if acid and Q[acid[1], 1] < Q[acid[2], 1]:                  # same rotamer for every acid: C=O up, OH down
        Q[[acid[1], acid[2]]] = Q[[acid[2], acid[1]]]
    for i in range(m.GetNumAtoms()):
        c.SetAtomPosition(i, (float(Q[i, 0]), float(Q[i, 1]), 0.0))
    Chem.Kekulize(m, clearAromaticFlags=True)                     # alternating bonds, not aromatic circles
    return Chem.MolToMolBlock(m)


def drawing_names(rows):
    """File name of each drawing: the first word of its label (letters and digits only), numbered when a panel repeats it."""
    stem = [re.sub(r"[^A-Za-z0-9]", "", r["label"].split()[0]) for r in rows]
    key = [(r["figure"], r["panel"], s) for r, s in zip(rows, stem)]
    total, seen, out = Counter(key), Counter(), []
    for r, s, k in zip(rows, stem, key):
        seen[k] += 1
        out.append(f"Fig{r['figure']}_{r['panel']}_{s}{seen[k] if total[k] > 1 else ''}")
    assert len(set(out)) == len(out)
    return out


def main(figure):
    import win32com.client
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(SPEC, encoding="utf-8")) if r["figure"] == str(figure)]
    assert rows, figure
    old = {r["name"]: r for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))} if MANIFEST.exists() else {}
    ctl = win32com.client.Dispatch("ChemDrawControl22.ChemDrawCtl")
    TMP = tempfile.TemporaryDirectory()
    s = ctl.Settings                                               # ACS Document 1996
    for k, v in dict(BondLength=14.4, LineWidth=0.6, BoldWidth=2.0, HashSpacing=2.5, MarginWidth=1.6, LabelSize=10.0, CaptionSize=10.0).items():
        setattr(s, k, v)                                           # the control's defaults follow the last style used in ChemDraw
    assert abs(s.BondLength - 14.4) < 0.01 and abs(s.LineWidth - 0.6) < 0.01 and s.LabelFont == "Arial" and s.LabelSize == 10
    keep = {k: v for k, v in old.items() if not k.startswith(f"Fig{figure}_")}
    out = []
    for r, name in zip(rows, drawing_names(rows)):
        cdxml, svg = OUT / f"{name}.cdxml", OUT / f"{name}.svg"
        prev = old.get(name)
        edited = bool(prev and cdxml.exists() and sha(cdxml) != prev["cdxml_sha256"]) or (prev or {}).get("source") == EDITED
        tc, ts = Path(TMP.name) / cdxml.name, Path(TMP.name) / svg.name   # the control cannot use a non-ASCII path
        if edited:
            shutil.copyfile(cdxml, tc)
        else:
            anchor, angle = ANCHOR[r["role"].split(" (")[0]]               # 'N-H partner (amide, BH|1, BH|2)' -> 'N-H partner'
            angle = PANEL_FACE.get((r["figure"], r["panel"]), angle)
            angle = ROLE_FACE.get((r["figure"], r["panel"], r["role"].split(" (")[0]), angle)
            ctl.Objects.Clear()
            ctl.Objects.SetData("chemical/x-mdl-molfile", oriented_molblock(r["smiles"], anchor, angle))
            ctl.SaveAs(str(tc))
            assert tc.exists() and tc.stat().st_size > 0, f"ChemDraw did not write {tc}"
            shutil.copyfile(tc, cdxml)
        drawn = Chem.MolsFromCDXMLFile(str(tc))                     # a hand edit may move atoms, never change the molecule
        assert len(drawn) == 1 and Chem.MolToSmiles(drawn[0]) == Chem.MolToSmiles(Chem.MolFromSmiles(r["smiles"])), name
        ctl.Objects.Clear()                                         # Open adds to the current drawing; start empty
        ctl.Open(str(tc), True)                                     # the SVG always comes from the CDXML
        ctl.SaveAs(str(ts))
        assert ts.exists() and ts.stat().st_size > 0, f"ChemDraw did not write {ts}"
        hetero = sorted(a.GetSymbol() for a in drawn[0].GetAtoms() if a.GetSymbol() != "C")
        labels = structure_art.load(ts)["labels"]                   # a subscript is its own, smaller label; H2N may come as H, 2, N
        symbols = sorted(re.findall(r"[A-Z][a-z]?", "".join(l["text"] for l in labels if l["scale"] == 1).replace("H", "")))
        assert symbols == hetero, f"{name}: atom labels {symbols}, heteroatoms {hetero}"
        shutil.copyfile(ts, svg)
        out.append(dict(name=name, panel=r["panel"], label=r["label"], role=r["role"], smiles=r["smiles"],
                        cdxml_sha256=sha(cdxml), svg_sha256=sha(svg), source=EDITED if edited else GENERATED))
        print("re-exported hand-edited" if edited else "drew", name)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "panel", "label", "role", "smiles", "cdxml_sha256", "svg_sha256", "source"])
        w.writeheader()
        w.writerows(list(keep.values()) + out)


if __name__ == "__main__":
    main(int(sys.argv[1]))

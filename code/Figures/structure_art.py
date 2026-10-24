"""ChemDraw drawings placed in matplotlib figures without redrawing the chemistry.

ChemDraw's SVG export holds each bond as a filled polygon and each atom label as Arial text at a fixed position. The
polygons are copied point for point; the labels are set in the same Arial at the same baseline position by matplotlib,
so structure labels and figure text share one embedded font. Every drawing in a figure is placed at one scale, fixed
by the label size.
"""
import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from matplotlib.patches import Polygon

NS = "{http://www.w3.org/2000/svg}"
PT_MM = 25.4 / 72


def _matrix(t):
    m = re.fullmatch(r"matrix\(([^)]*)\)", t.strip())
    a, b, c, d, e, f = (float(x) for x in m.group(1).replace(",", " ").split())
    return np.array([[a, c, e], [b, d, f]])


def _polygon(d, M):
    tok = re.findall(r"[MLZ]|-?[\d.]+(?:e-?\d+)?", d)
    pts, i = [], 0
    while i < len(tok):
        t = tok[i]
        if t in "ML":
            pts.append((float(tok[i + 1]), float(tok[i + 2]))); i += 3
        elif t == "Z":
            i += 1
        else:
            raise ValueError(f"unexpected path token {t!r}")
    P = np.c_[np.array(pts), np.ones(len(pts))]
    return (M @ P.T).T


def load(svg):
    """Polygons, labels and size (SVG px) of one ChemDraw drawing; anything ChemDraw might add later fails loudly."""
    root = ET.parse(svg).getroot()
    w, h = (float(root.get(k).removesuffix("px")) for k in ("width", "height"))
    polys, labels = [], []
    for el in root:
        tag = el.tag.removeprefix(NS)
        if tag == "path":
            assert el.get("fill") == "#000000" and el.get("stroke") == "none", el.attrib
            assert set(re.findall(r"[A-Za-z]", el.get("d"))) <= set("MLZe"), el.get("d")
            polys.append(_polygon(el.get("d"), _matrix(el.get("transform"))))
        elif tag == "text":
            assert len(el) == 0, "labels with nested elements need their own handling"
            assert el.get("font-family") == "Arial" and el.get("font-style") == "normal" and el.get("font-weight") == "normal"
            M = _matrix(el.get("transform"))
            assert abs(M[0, 0] - M[1, 1]) < 1e-9 and M[0, 1] == M[1, 0] == 0
            size = float(el.get("font-size").removesuffix("px")) * M[0, 0]
            labels.append(dict(text=el.text.strip(), x=M[0, 2], y=M[1, 2], size=size))
        else:
            raise ValueError(f"unexpected SVG element {tag}")
    assert labels, "a drawing without atom labels"
    main = max(l["size"] for l in labels)         # atom labels; ChemDraw writes a subscript (the 2 of NH2) as its own, smaller label
    for l in labels:
        l["scale"] = l["size"] / main
        assert abs(l["scale"] - 1) < 1e-6 or (l["text"].isdigit() and abs(l["scale"] - 0.75) < 1e-6), l
        l["scale"] = 1.0 if abs(l["scale"] - 1) < 1e-6 else 0.75
    return dict(path=str(svg), sha256=hashlib.sha256(Path(svg).read_bytes()).hexdigest(), width=w, height=h,
                polygons=polys, labels=labels, label_px=round(main, 6))


def size_mm(art, label_pt):
    k = label_pt / art["label_px"] * PT_MM
    return art["width"] * k, art["height"] * k


def draw(ax, art, x_mm, y_mm, label_pt, gid):
    """Draw on an axes whose data units are mm with y pointing down; (x_mm, y_mm) is the drawing's top-left corner."""
    k = label_pt / art["label_px"] * PT_MM
    for i, P in enumerate(art["polygons"]):
        ax.add_patch(Polygon(P * k + (x_mm, y_mm), closed=True, facecolor="black", edgecolor="none", lw=0,
                             gid=f"{gid}|bond{i}", zorder=5))
    for i, l in enumerate(art["labels"]):
        ax.text(x_mm + l["x"] * k, y_mm + l["y"] * k, l["text"], fontsize=label_pt * l["scale"], ha="left", va="baseline",
                color="black", gid=f"{gid}|label{i}", zorder=5)
    return size_mm(art, label_pt)

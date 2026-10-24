"""Shared pieces of the Extended Data figures: the public rows, the arrays in Fig. 2A's order, drawing in mm on a 180 mm
page, the read-back of what was drawn, and the writing of a figure with its Source Data and provenance.

Every Extended Data figure follows the rules of the current main figures 1-5: values are recomputed from the public measurements and must
agree with the frozen outputs before anything is drawn; the Source Data hold only what is drawn or quoted in the legend;
outputs are a vector PDF, an SVG and a 1,200 dpi PNG.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt                                              # noqa: E402
from matplotlib.colors import to_hex                                         # noqa: E402
from matplotlib.patches import Rectangle                                     # noqa: E402
from matplotlib.text import Text                                             # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path[:0] = [str(ROOT / "06_CODE/Core/00_common"), str(ROOT / "06_CODE/Core/00_common/ctb")]
import fig_style as fs                                                       # noqa: E402
from figure1_data import public_units, write_xlsx                            # noqa: E402,F401
import anatomy                                                               # noqa: E402

DATA = ROOT / "05_DATA"
RAW = DATA / "Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz"
ANATOMY_ARRAYS = DATA / "Results/display_inputs/Anatomy_arrays.csv"
# read by the frozen fold accessor (boundaries.verify_file) when it checks the registry against its lock
REGISTRY_INTEGRITY = (DATA / "Analysis/06_METADATA/input_lock.json", ROOT / "07_METHODS_AND_PROVENANCE/analysis_runtime/PATH_RELOCATIONS.json")
FIGDIR, SDDIR, QCDIR = ROOT / "02_FIGURES/Supplementary_Figures", ROOT / "04_SOURCE_DATA/Supplementary_Figures", ROOT / "08_REVIEW_AND_QC/extended_data"
W = fs.WIDTH_MM
LEFT, RIGHT = 0.15, fs.WIDTH_MM - 0.5                                        # text starts and right edges, inside the page
PNG_DPI = 1200
TOL = 1e-9
sha_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rel = lambda p: Path(p).resolve().relative_to(ROOT).as_posix()


def paths(n, out=None, extra=()):
    """Delivered files of Fig. Sn (and, with out, where a scratch build writes them)."""
    base = dict(pdf=FIGDIR / f"FigS{n}.pdf", svg=FIGDIR / f"FigS{n}.svg", png=FIGDIR / f"FigS{n}.png",
                xlsx=SDDIR / f"FigS{n}_SourceData.xlsx")
    base.update({k: SDDIR / name for k, name in extra})
    base.update(provenance=SDDIR / f"FigS{n}_SourceData_provenance.json", readback=QCDIR / f"ed{n}_plot_readback.json")
    return base, {k: (Path(out) / v.name if out else v) for k, v in base.items()}


def owner_roles():
    return pd.read_csv(ANATOMY_ARRAYS).set_index("screen").owner_role.astype(int).to_dict()


def order():
    """Fig. 2A's order (arrays with an electrophilic halide role first, alphabetical by display name within each group)."""
    hal, other = fs.array_order({k: anatomy.electrophile_role(k) is not None for k in fs.ARRAY_LABEL})
    return hal + other, {k: ("electrophilic halide" if k in hal else "other") for k in hal + other}


def figure(H, salt):
    fs.apply(); fs.check_font()
    plt.rcParams["svg.hashsalt"] = salt
    fig = plt.figure(figsize=(W * fs.MM, H * fs.MM))
    cv = fig.add_axes([0, 0, 1, 1]); cv.set_xlim(0, W); cv.set_ylim(H, 0); cv.axis("off"); cv.set_gid("canvas")
    return fig, cv


def heading(fig, cv, x, y, letter, title, H, note=None):
    """Panel letter (if any), its title on the same line and, below them, one grey note."""
    if letter:
        fs.letter(fig, x, y, letter, H).set_gid(f"letter|{letter}")
    fs.title(fig, x + (4.2 if letter else LEFT), y + 0.35, title, H).set_gid(f"title|{letter or 'main'}")
    if note:
        cv.text(x + (4.2 if letter else LEFT), y + 5.8, note, gid=f"note|{letter or 'main'}", ha="left", va="center",
                fontsize=fs.PT["tiny"], color=fs.C["muted"])


def clean(ax, keep=("left", "bottom")):
    for sp_ in ("left", "right", "top", "bottom"):
        ax.spines[sp_].set_visible(sp_ in keep)
    ax.patch.set_visible(False)


def readback(fig, H):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px_mm = 25.4 / fig.dpi

    def bbox(a):
        e = a.get_window_extent(rend)
        return [round(e.x0 * px_mm, 3), round(H - e.y1 * px_mm, 3), round(e.x1 * px_mm, 3), round(H - e.y0 * px_mm, 3)]
    out = dict(width_mm=W, height_mm=H, rectangles=[], markers=[], lines=[], texts=[], axes={})
    for ax in fig.axes:
        if ax.get_gid() and ax.get_gid() != "canvas":
            out["axes"][ax.get_gid()] = dict(bbox_mm=bbox(ax), xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()))
        for p in ax.patches:
            if p.get_gid() and isinstance(p, Rectangle):
                out["rectangles"].append(dict(gid=p.get_gid(), axes=ax.get_gid(), x=float(p.get_x()), y=float(p.get_y()), width=float(p.get_width()),
                                              height=float(p.get_height()), color=to_hex(p.get_facecolor()), bbox_mm=bbox(p)))
        for ln in ax.lines:
            if ln.get_gid():
                xy = ln.get_transform().transform(np.c_[np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)])
                rec = dict(gid=ln.get_gid(), axes=ax.get_gid(), x=np.asarray(ln.get_xdata(), float).tolist(), y=np.asarray(ln.get_ydata(), float).tolist(),
                           color=to_hex(ln.get_color()), bbox_mm=bbox(ln), width_pt=ln.get_linewidth(),
                           path_mm=[[round(px * px_mm, 3), round(H - py * px_mm, 3)] for px, py in xy])
                if ln.get_marker() not in (None, "None", ""):
                    rec.update(marker=ln.get_marker(), face=to_hex(ln.get_markerfacecolor()), size=ln.get_markersize())
                    out["markers"].append(rec)
                else:
                    out["lines"].append(rec)
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            out["texts"].append(dict(gid=t.get_gid(), text=t.get_text(), size=t.get_fontsize(), weight=str(t.get_fontweight()), bbox_mm=bbox(t)))
    return out


def write(n, fig, H, sheets, prov, out=None, extra=(), tables=()):
    """Save Fig. Sn: figure files, read-back, Source Data workbook, any derived CSV files (tables: (key,
    DataFrame)) and the provenance, which records the hashes of everything written."""
    base, p = paths(n, out, extra)
    for q in p.values():
        q.parent.mkdir(parents=True, exist_ok=True)
    rb = readback(fig, H)
    fig.savefig(p["pdf"], metadata={"Creator": None, "Producer": None, "CreationDate": None})
    fig.savefig(p["svg"], metadata={"Creator": None, "Date": None})
    fig.savefig(p["png"], dpi=PNG_DPI, metadata={"Software": None})
    plt.close(fig)
    p["readback"].write_text(json.dumps(rb, indent=1, ensure_ascii=False), encoding="utf-8")
    write_xlsx(sheets, p["xlsx"])
    for key, t in tables:
        t.to_csv(p[key], index=False, lineterminator="\n")
        assert pd.read_csv(p[key], float_precision="round_trip").equals(t), f"{p[key].name} does not read back exactly"
    prov = dict(prov, outputs={k: dict(file=rel(base[k]), sha256=sha_file(q)) for k, q in p.items() if k != "provenance"})
    p["provenance"].write_text(json.dumps(prov, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return p

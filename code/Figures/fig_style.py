"""Shared visual language of the directional figures (Nature): Arial, 5-8 pt, thin lines, 180 mm width.

Array names and orders are defined once here so that every figure lists arrays the same way; orders never depend
on an outcome.
"""
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.text as mtext

ROOT = Path(__file__).resolve().parents[2]
MM = 1 / 25.4
WIDTH_MM = 180.0
FONT = "Arial"
PT = dict(letter=8.0, title=6.5, base=6.0, small=5.5, tiny=5.0)
LW = dict(axis=0.5, tick=0.5, thin=0.4, bar=0.0)
C = dict(text="#1A1A1A", muted="#6B7380", faint="#B9BEC5", grid="#E6E8EB",
         owner="#1F3B5A", partner="#A3B6CA", molecular="#1F3B5A", combination="#AEB4BC", noise="#D9DCE0",
         control="#8A9099", band="#EEF2F6", keepA="#E3EBF3", keepB="#F3ECE1", accent="#9C4A28")

ARRAY_LABEL = {   # the names of table S1, which holds the same per-array values
    "Acylation|1": "Amide coupling", "Buchwald-Hartwig|1": "C–N (Pd ligand)", "Buchwald-Hartwig|2": "C–N (Pd/Cu/Ni)",
    "C-O competitive reactions|1": "C–O (Pd/Cu)", "Chan-Lam|1": "Chan–Lam (Cu)", "Esterification|1": "Esterification",
    "Grignard|1": "Grignard addition", "Heck|1": "Heck (solvent/base)", "Hiyama|1": "Hiyama (Pd)",
    "Kumada|1": "Kumada (Pd/Ni/Fe)", "Reductive-amination|1": "Reductive amination", "SNAr|1": "SNAr (base/solvent)",
    "Sonogashira|1": "Sonogashira (solvent/base)", "Suzuki|1": "Suzuki (Pd)", "Ullmann|1": "Ullmann C–O (Cu)",
}


def array_order(electrophile_flag):
    """Arrays with an electrophilic halide role first, then the others; alphabetical by display name within each group."""
    hal = sorted([k for k, v in electrophile_flag.items() if v], key=lambda k: ARRAY_LABEL[k].casefold())
    other = sorted([k for k, v in electrophile_flag.items() if not v], key=lambda k: ARRAY_LABEL[k].casefold())
    return hal, other


def _subscript_snar():
    """Every text drawn on a figure prints SNAr with a subscript N (S with upright subscript N, followed by Ar); Source Data keep the plain name."""
    if getattr(mtext.Text, "_snar_subscript", False):
        return
    base = mtext.Text.set_text

    def set_text(self, s):
        if isinstance(s, str) and "SNAr" in s:
            s = re.sub(r"\bSNAr\b", lambda _m: r"S$_{\mathrm{N}}$Ar", s)
        return base(self, s)
    mtext.Text.set_text, mtext.Text._snar_subscript = set_text, True


def apply():
    _subscript_snar()
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT], "font.size": PT["base"],
        "axes.linewidth": LW["axis"], "axes.labelsize": PT["base"], "axes.titlesize": PT["title"],
        "axes.edgecolor": C["text"], "axes.labelcolor": C["text"], "text.color": C["text"],
        "xtick.labelsize": PT["small"], "ytick.labelsize": PT["small"], "xtick.major.width": LW["tick"],
        "ytick.major.width": LW["tick"], "xtick.major.size": 2.0, "ytick.major.size": 2.0,
        "xtick.color": C["text"], "ytick.color": C["text"], "legend.fontsize": PT["small"], "legend.frameon": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "mathtext.fontset": "custom", "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold", "axes.unicode_minus": True,
    })


def check_font():
    """The figures need regular, bold and italic Arial; a silent fallback would change every glyph."""
    from matplotlib import font_manager as fm
    have = {(f.style, f.weight) for f in fm.fontManager.ttflist if f.name == FONT}
    need = {("normal", 400), ("normal", 700), ("italic", 400)}
    missing = need - have
    assert not missing, f"{FONT} variants missing: {missing}"


def letter(fig, x_mm, y_mm, s, H_mm):
    """Panel letter at (x, y) measured from the top-left corner of the figure, in mm."""
    x_mm = max(x_mm, 0.5)                                  # a capital letter has no side bearing: keep 0.5 mm of white
    return fig.text(x_mm / WIDTH_MM, 1 - y_mm / H_mm, s.upper(), fontsize=PT["letter"], fontweight="bold", va="top", ha="left")   # Science: capital panel letters


def title(fig, x_mm, y_mm, s, H_mm):
    return fig.text(x_mm / WIDTH_MM, 1 - y_mm / H_mm, s, fontsize=PT["title"], va="top", ha="left")


def axes_mm(fig, x_mm, y_mm, w_mm, h_mm, H_mm, **kw):
    """Axes placed by its top-left corner and size in mm; no constrained layout, so positions stay where they are put."""
    return fig.add_axes([x_mm / WIDTH_MM, 1 - (y_mm + h_mm) / H_mm, w_mm / WIDTH_MM, h_mm / H_mm], **kw)

"""Four-figure presentation, named retained inputs and deterministic exports."""
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

import structure_art
from figure_export import InputTrace, capture_readback, write_provenance
from figure1_data import write_xlsx
from figure3_selection import load_art

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "02_FIGURES/Main"
DATA = ROOT / "04_SOURCE_DATA"
REFERENCE = ROOT / "05_DATA/Analysis/figure_display/v5_reference_tables"
REFERENCE_NAMES = {1: "overview", 2: "preference", 3: "screen_shares", 4: "molecule_context", 5: "transfer"}
BLUE, ORANGE = "#0072B2", "#D55E00"
DARK, LIGHT = "#4D4D4D", "#B3B3B3"
HEAT = LinearSegmentedColormap.from_list("v5_preference", ["#5A5A5A", "#FFFFFF", "#D9B748"])
WIDTH = 180.0
OVERRIDE = None


@contextmanager
def style():
    with plt.rc_context({
        "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "axes.titleweight": "normal", "axes.linewidth": .5,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.width": .5, "ytick.major.width": .5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "p2-v5-draft",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.grid": False,
    }):
        yield


def axes_mm(fig, x, y, w, h):
    height = fig.get_figheight() * 25.4
    return fig.add_axes([x / WIDTH, 1 - (y + h) / height, w / WIDTH, h / height])


def new_figure(height):
    assert 0 < height <= 215
    fig = plt.figure(figsize=(WIDTH / 25.4, height / 25.4), dpi=160)
    canvas = axes_mm(fig, 0, 0, WIDTH, height)
    canvas.set(xlim=(0, WIDTH), ylim=(height, 0))
    canvas.axis("off")
    return fig, canvas


def panel(canvas, letter, x, y, tag=None):
    canvas.text(x, y, letter, fontsize=9, weight="bold", va="center", gid=f"panel_{letter}")
    if tag:
        assert len(tag.split()) <= 5
        canvas.text(x + 5, y, tag, fontsize=8, va="center", gid=f"tag_{letter}")


def structure(canvas, path, x, y, gid):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    art = load_art(path)
    size = structure_art.draw(canvas, art, x, y, 8, gid)
    for patch in canvas.patches:
        if (patch.get_gid() or "").startswith(gid + "|bond"):
            patch.set_edgecolor("black")
            patch.set_linewidth(.02)
    return size


def structure_size(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return structure_art.size_mm(load_art(path), 8)


def reference_path(number):
    return REFERENCE / f"{REFERENCE_NAMES[number]}.xlsx"


def source_path(value):
    if not isinstance(value, str):
        return value
    for number in (1, 2, 3, 4, 5):
        value = value.replace(f"04_SOURCE_DATA/Figure{number}/Figure{number}_SourceData.xlsx",
                              reference_path(number).relative_to(ROOT).as_posix())
    value = value.replace("04_SOURCE_DATA/Figure2/figure2_payload.json",
                          "05_DATA/Analysis/figure_display/v5_reference_tables/preference_payload.json")
    value = value.replace("04_SOURCE_DATA/Figure5/", "04_SOURCE_DATA/Figure4/")
    value = value.replace("06_CODE/Figures/figure5.py", "06_CODE/Figures/figure4.py")
    return value


def read_tables(number):
    """Read named pre-migration computed inputs, never this run's outputs."""
    path = reference_path(number)
    records = json.loads((REFERENCE / "MANIFEST.json").read_text(encoding="utf-8"))
    record = next(row for row in records if row["reference"] == path.relative_to(ROOT).as_posix())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    return pd.read_excel(path, sheet_name=None, dtype=object, keep_default_na=False, na_values=[""])


def load_payload():
    return json.loads((REFERENCE / "preference_payload.json").read_text(encoding="utf-8"))


def write_json(number, filename, value):
    target = (OVERRIDE or DATA / f"Figure{number}") / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def save(fig, number, tables, extra=None):
    from matplotlib.text import Text
    from figure_layout_checks import d21

    output = OVERRIDE or ART
    sources = OVERRIDE or DATA / f"Figure{number}"
    output.mkdir(parents=True, exist_ok=True)
    sources.mkdir(parents=True, exist_ok=True)
    clearance = d21(fig, number)
    assert clearance["status"] == "PASS", f"Figure {number}: D21 spacing failure: {clearance['failures']}"
    metadata = {"Creator": "P2 main figures", "CreationDate": None, "ModDate": None}
    fig.savefig(output / f"Figure{number}.pdf", metadata=metadata)
    fig.savefig(output / f"Figure{number}.svg", metadata={"Date": None, "Creator": "P2 main figures"})
    fig.savefig(output / f"Figure{number}.png", dpi=600)
    tables = {key: frame.copy() for key, frame in tables.items()}
    if "Printed_numbers" in tables:
        tables["Retained_v3_numbers"] = tables.pop("Printed_numbers")
    if "V5_printed_numbers" in tables:
        tables["Printed_numbers"] = tables.pop("V5_printed_numbers")
    for frame in tables.values():
        for column in frame.columns:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(source_path)
    write_xlsx(tables, sources / f"Figure{number}_SourceData.xlsx")
    titles = {t.get_gid().replace("tag_", ""): t.get_text()
              for t in fig.findobj(Text) if (t.get_gid() or "").startswith("tag_")}
    readback = capture_readback(fig, number, titles, extra or {})
    readback["d21"] = clearance
    readback["text_styles"] = [dict(text=t.get_text(), gid=t.get_gid(),
                                    weight=t.get_fontweight(), color=t.get_color())
                               for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    (sources / f"Figure{number}_readback.json").write_text(
        json.dumps(readback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    return {ext: str(output / f"Figure{number}.{ext}") for ext in ("pdf", "svg", "png")}


def cli(main, number):
    global OVERRIDE
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    OVERRIDE = args.out
    panels, outputs = {}, {}
    with InputTrace() as trace:
        main()
        sidecars = {1: "figure1_payload.json", 2: "figure2_selection.json", 4: "figure4_selection.json"}
        if number in sidecars:
            name = sidecars[number]
            logical = DATA / f"Figure{number}" / name
            outputs["payload" if number == 1 else "selection"] = (logical, (args.out or logical.parent) / name)
        if number == 2:
            from figure1_data import public_units, noise_scale
            agreement, scale, pairs, pearson = noise_scale(*public_units())
            panels = {"A": {"noise_calibration": dict(duplicate_agreement=agreement,
                      model_scale=scale, pairs=pairs, duplicate_pattern_pearson=pearson)}}
        if number == 4:
            from figure4_bootstrap_data import write_seta_bootstrap
            paths = write_seta_bootstrap(args.out)
            outputs.update({name: (entry["logical"], entry["actual"]) for name, entry in paths.items()})
    write_provenance(number, trace.paths, actual_dir=args.out, outputs=outputs, panels=panels)

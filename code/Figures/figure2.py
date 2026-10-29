"""Figure 2: approved v6 screen comparisons and external direction checks."""
from __future__ import annotations

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "06_CODE/Figures"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import pymupdf as fitz

import main_figure_common as V
import v6_core as c
from external_figure_input import load_plot_input, EXPECTED_SHA256
from types import SimpleNamespace
from figure3_selection import load_art, verify_rule
import structure_art

WIDTH, HEIGHT = 180.0, 215.0
FONT, ATOM, PANEL = 7.0, 7.0, 8.0
RESULT = ROOT / "05_DATA/Results/external_replication/Figure2_plot_input.json"
RESULT_SHA = EXPECTED_SHA256
CHEM = ROOT / "02_FIGURES/ChemDraw/v3/f3"
PT_MM = 25.4 / 72
F=SimpleNamespace(
    ORDER=["Buchwald-Hartwig|1","Buchwald-Hartwig|2","C-O competitive reactions|1","Heck|1","Hiyama|1",
           "Kumada|1","SNAr|1","Sonogashira|1","Suzuki|1","Ullmann|1","Acylation|1","Chan-Lam|1",
           "Esterification|1","Grignard|1","Reductive-amination|1"],
    ADDITIVES=["AcOH","MsOH","PTSA","TFA","none","DIPEA","TEA","2,6-lutidine","Py","DBU",
               "Cs2CO3","K2CO3","KHCO3","KOAc","KOH","KOtBu","NaOMe","NaOtAm"],
    SOLVENTS=["DMSO","THF","t-AmOH","NMP","CPME","MeCN"],
    REACTION_LABELS={"Buchwald-Hartwig|1":"C\u2013N (Pd ligand)","Buchwald-Hartwig|2":"C\u2013N (Pd/Cu/Ni)",
                    "C-O competitive reactions|1":"C\u2013O (Pd/Cu)","Heck|1":"Heck","Hiyama|1":"Hiyama (Pd)",
                    "Kumada|1":"Kumada (Pd/Ni/Fe)","SNAr|1":r"S$_{\mathrm{N}}$Ar","Sonogashira|1":"Sonogashira",
                    "Suzuki|1":"Suzuki (Pd)","Ullmann|1":"Ullmann C\u2013O (Cu)","Acylation|1":"Amide coupling",
                    "Chan-Lam|1":"Chan\u2013Lam (Cu)","Esterification|1":"Esterification",
                    "Grignard|1":"Grignard addition","Reductive-amination|1":"Reductive amination"})


def _role_colour(role,owner):
    role=str(role).replace("\u2013","-")
    if "halide"in role:return V.BLUE
    if role in("N-H partner","N-H nucleophile","amine"):return V.ORANGE
    return V.DARK if owner else V.LIGHT


F._role_colour=_role_colour


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    return value


def label(canvas, x, y, text, gid, **kw):
    kw.setdefault("ha", "left")
    return canvas.text(x, y, text, fontsize=FONT, va="center", gid=gid, **kw)


def panel(canvas, letter, x, y, title):
    canvas.text(x, y, letter, fontsize=PANEL, weight="bold", style="normal", va="center", gid="panel_" + letter)
    label(canvas, x + 5, y, title, "title_" + letter)


def art_bounds(canvas, artists, renderer):
    inverse = canvas.transData.inverted()
    corners = []
    for item in artists:
        if not isinstance(item, (Text, Polygon)):
            continue
        points = inverse.transform(item.get_window_extent(renderer).get_points())
        if isinstance(item, Polygon):
            pad = item.get_linewidth() * PT_MM / 2
            points = np.asarray([points.min(axis=0) - pad, points.max(axis=0) + pad])
        corners.append(points)
    assert corners
    points = np.concatenate(corners)
    return [*points.min(axis=0), *points.max(axis=0)]


def draw_structure(canvas, art, x, y, gid):
    result = structure_art.draw(canvas, art, x, y, ATOM, gid)
    for patch in canvas.patches:
        if (patch.get_gid() or "").startswith(gid + "|bond"):
            patch.set_edgecolor("black")
            patch.set_linewidth(.08)
    return result


def measure_structures(fig, canvas, arts):
    groups = []
    for i, art in enumerate(arts):
        previous = set(canvas.get_children())
        draw_structure(canvas, art, 0, 0, f"measure_{i}")
        groups.append([item for item in canvas.get_children() if item not in previous])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    result = [art_bounds(canvas, group, renderer) for group in groups]
    for group in groups:
        for item in group:
            item.remove()
    return result


def swarm(values, height_mm, value_range, diameter_mm=1.6):
    positions, offsets = [], np.zeros(len(values))
    choices = [0, 1.7, -1.7, 3.4, -3.4, 5.1, -5.1, 6.8, -6.8]
    for i in sorted(range(len(values)), key=lambda j: (values[j], j)):
        y_mm = (values[i] - value_range[0]) / (value_range[1] - value_range[0]) * height_mm
        chosen = next((x for x in choices if all((x - px) ** 2 + (y_mm - py) ** 2 >= diameter_mm ** 2
                                                for px, py in positions)), None)
        assert chosen is not None
        positions.append((chosen, y_mm))
        offsets[i] = chosen
    return offsets


def source_data():
    tables = V.read_tables(3)
    arrays, selected, conditions = (tables[k].copy() for k in ("Fig3A_arrays", "Rule2_selected", "Fig3B_conditions"))
    assert arrays.screen.tolist() == F.ORDER == selected.screen.tolist()
    assert len(arrays) == 15 and int(arrays.halide_array.sum()) == 10
    assert arrays.halide_array.astype(bool).tolist() == [True] * 10 + [False] * 5
    keys = ["screen", "smiles", "owner_role", "threshold_used", "product_forming_partners"]
    assert arrays[keys].to_dict("records") == selected[keys].to_dict("records")
    assert len(conditions) == conditions.condition_id.nunique() == 47
    assert len(conditions[["matrix_row", "matrix_column"]].drop_duplicates()) == 47
    assert (arrays.owner_share.astype(float) > arrays.partner_share.astype(float)).all()
    assert np.allclose(arrays.owner_fraction.astype(float), arrays.owner_share.astype(float) / arrays.share.astype(float), atol=1e-12)
    data,result_path,result_sha=load_plot_input(c.EXTERNAL_RESULT)
    assert result_sha==RESULT_SHA
    rule_sha = verify_rule()
    masters = pd.read_csv(CHEM / "structures_manifest.csv").set_index("name")
    inputs = [ROOT / "05_DATA/Analysis/figure_display/v5_reference_tables/screen_shares.xlsx",
              ROOT / "05_DATA/Analysis/figure_display/v5_reference_tables/MANIFEST.json", result_path,
              ROOT / "07_METHODS_AND_PROVENANCE/frozen_analysis_plans/figure_selection_v3/RULE.md",
              CHEM / "structures_manifest.csv"]
    inputs += [ROOT / "06_CODE/Figures" / name for name in
               ("figure2.py", "figure2_data.py", "main_figure_common.py", "external_figure_input.py", "structure_art.py", "figure3_selection.py")]
    arts = []
    for name in arrays.structure_file:
        svg = CHEM / name
        cdxml = svg.with_suffix(".cdxml")
        record = masters.loc[svg.stem]
        assert sha(svg) == record.svg_sha256 and sha(cdxml) == record.cdxml_sha256
        mother = ET.parse(cdxml).getroot()
        assert float(mother.get("BondLength")) == 14.4 and float(mother.get("LabelSize")) == 10
        assert mother.get("CreationProgram", "").startswith("ChemDraw")
        inputs.extend((svg, cdxml))
        arts.append(load_art(svg))
    return arrays, conditions, data, arts, inputs, rule_sha


def breadth(fig, canvas, arrays, arts):
    panel(canvas, "a", 1, 3.5, "Halide: 10 / 10")
    canvas.text(6, 3.5, "Halide: 10 / 10", fontsize=FONT, weight="bold", va="center", gid="a_key_conclusion")
    next(item for item in canvas.texts if item.get_gid() == "title_a").remove()
    legend = [(1, V.BLUE, "halide"), (22, V.ORANGE, "N-H partner"),
              (55, V.DARK, "other owner"), (88, V.LIGHT, "other partner")]
    for i, (x, color, text) in enumerate(legend):
        canvas.plot([x, x + 3.2], [13.0, 13.0], color=color, lw=2.0, solid_capstyle="butt", gid=f"a_role_key_{i}")
        label(canvas, x + 4.2, 13.0, text, f"a_role_legend_{i}")
    bounds = measure_structures(fig, canvas, arts)
    heights = np.maximum([box[3] - box[1] + 1.0 for box in bounds], 9.2)
    edges = 15.5 + np.r_[0, np.cumsum(heights + .2)]
    edges[10:] += 2.5
    centers = [edges[i] + heights[i] / 2 for i in range(15)]
    bottom = centers[-1] + heights[-1] / 2 + .5
    assert bottom < 205, bottom
    ax = V.axes_mm(fig, 81, edges[0], 40, bottom - edges[0])
    ax.set(xlim=(-8, 52), ylim=(bottom, edges[0]), xticks=[0, 20, 40], yticks=[])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.axvline(0, color="#B3B3B3", lw=.5, zorder=0)
    ax.set_xlabel("Preference share (%)", labelpad=4, fontsize=FONT)
    ax.set_gid("a_share_axes")
    canvas.plot([1, 77], [edges[10] - 1.4] * 2, color="#B3B3B3", lw=.5, gid="a_halide_separator")
    layout = []
    for i, (row, center, box, art) in enumerate(zip(arrays.itertuples(), centers, bounds, arts), 1):
        if row.screen == "SNAr|1":
            tokens = [("S", FONT, 0), ("N", 5.25, .55), ("Ar", FONT, 0)]
            x = 1.0
            renderer = fig.canvas.get_renderer()
            for j, (text, size, lower) in enumerate(tokens):
                canvas.text(x, center - 1.6 + lower, text, fontsize=size, fontstyle="normal",
                            ha="left", va="center", gid=f"a_reaction_{i}" + (f"_part_{j}" if j else ""))
                x += renderer.get_text_width_height_descent(
                    text, FontProperties(family="Arial", size=size), False)[0] * 25.4 / fig.dpi
            label(canvas, 1, center + 1.8, f"{100 * row.owner_share:.1f}% / {100 * row.partner_share:.1f}%", "a_snar_shares")
        else:
            text = F.REACTION_LABELS[row.screen] + ("*" if bool(row.aryl_bromide_only) else "")
            label(canvas, 1, center, text, f"a_reaction_{i}")
        x, y = 56.7 - (box[0] + box[2]) / 2, center - (box[1] + box[3]) / 2
        w, h = draw_structure(canvas, art, x, y, f"a_structure_{i}")
        for owner, offset, role in ((True, -1.0, int(row.owner_role)), (False, 1.0, 3 - int(row.owner_role))):
            value = 100 * float(getattr(row, f"share{role}"))
            ax.barh(center + offset, value, height=1.45,
                    color=F._role_colour(getattr(row, f"role{role}"), owner), edgecolor="none",
                    gid=f"a_{'owner' if owner else 'partner'}_{i}")
        layout.append(dict(screen=row.screen, structure=str(CHEM / row.structure_file),
                           x_mm=x, y_mm=y, center_y_mm=center, width_mm=w, height_mm=h,
                           row_top_mm=edges[i - 1], row_bottom_mm=edges[i - 1] + heights[i - 1],
                           owner_share_pct=100 * row.owner_share, partner_share_pct=100 * row.partner_share))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for i, row in enumerate(layout, 1):
        row["visible_bounds_mm"] = art_bounds(canvas, [item for item in canvas.get_children()
            if (item.get_gid() or "").startswith(f"a_structure_{i}|")], renderer)
    return layout


def summary(fig, canvas, arrays):
    panel(canvas, "b", 125, 3.5, "One dominant molecule")
    ax = V.axes_mm(fig, 135, 20, 43, 62)
    limit = 150.0
    ax.set(xlim=(-.5, 1.5), ylim=(0, limit), xticks=[], yticks=[0, 50, 100, 150])
    ax.spines[["right", "top", "bottom"]].set_visible(False)
    ax.set_gid("b_common_percent_axes")
    label(canvas, 131, 14, "%", "b_percent_unit")
    rows = []
    for i, (field, denominator) in enumerate((("share", "total preference"), ("owner_fraction", "molecular share"))):
        values = 100 * arrays[field].to_numpy(float)
        display_values = np.minimum(values, limit)
        offsets = swarm(display_values, 62, (0, limit)) / (43 / 2)
        visible = values <= limit
        ax.scatter(i + offsets[visible], values[visible], s=13, c=V.DARK, linewidths=.3,
                   edgecolors="black", gid="b_points_" + field)
        for j in np.flatnonzero(~visible):
            ax.scatter(i + offsets[j], limit, s=15, marker="^", c=V.DARK, linewidths=.3,
                       edgecolors="black", clip_on=False, gid="b_points_" + field + "_overflow")
            x = 135 + (i + offsets[j] + .5) / 2 * 43
            label(canvas, x + 2.9, 20, f"{values[j]:.1f}%", "b_overflow_label_" + field)
        median = float(np.median(values))
        ax.plot([i - .22, i + .22], [median, median], color="black", lw=1.0, gid="b_median_" + field)
        x = 135 + (i + .5) / 2 * 43
        label(canvas, x, 13, f"Median {median:.0f}%", "b_median_label_" + field, ha="center")
        bottom = "Molecules / total\npreference" if i == 0 else "Owner / molecular\nshare"
        label(canvas, x, 88, bottom, "b_denominator_" + field, ha="center", linespacing=1.2)
        rows.extend(dict(screen=screen, quantity=field, denominator=denominator, value_pct=value,
                         display_value_pct=display_value, display_x_offset=offset,
                         overflow=bool(value > limit),
                         display_marker="triangle" if value > limit else "circle",
                         displayed_annotation=f"{value:.1f}%" if value > limit else None)
                    for screen, value, display_value, offset in zip(arrays.screen, values, display_values, offsets))
    label(canvas, 156.5, 96, "15 screens", "b_denominator_count", ha="center")
    return rows


def external(fig, canvas, data):
    panel(canvas, "c", 125, 103.0, "Same direction externally")
    rows, stats = [], []
    for index, (name, title, reference, roles) in enumerate((
            ("CH_arylation", "C-H arylation", "BH_ligand", ("Aryl bromide", "Imidazole")),
            ("Suzuki", "Suzuki-Miyaura", "Suzuki", ("Aryl halide", "Boron partner")))):
        offset = 55 * index
        label(canvas, 134, 115 + offset, title, "c_reaction_" + name)
        ax = V.axes_mm(fig, 134, 123 + offset, 42, 26)
        ax.set(xlim=(-.5, 1.5), ylim=(-1, 1), xticks=[], yticks=[-1, 0, 1])
        ax.spines[["right", "top", "bottom"]].set_visible(False)
        ax.axhline(0, color="#D0D0D0", lw=.5, zorder=0)
        ax.set_gid("c_axes_" + name)
        if index == 0:
            ax.set_ylabel("Molecular agreement", fontsize=FONT, labelpad=4)
        point = data["datasets"][name]["main"]["estimates"] if index == 0 else data["datasets"][name]["point"]
        ci = data["datasets"][name]["consistency_bootstrap" if index == 0 else "bootstrap"]["difference_ci95"]
        for role_index, role in enumerate(("halide", "partner")):
            eligible = [r for r in data["molecules"][name] if r["role"] == role and r["status"] == "eligible"]
            values = np.asarray([r["consistency"] for r in eligible], dtype=float)
            median = float(point[role + "_consistency"])
            assert abs(float(np.median(values)) - median) < 1e-12
            color = V.BLUE if role == "halide" else V.ORANGE if index == 0 else V.LIGHT
            offsets = swarm(values, 26, (-1, 1), diameter_mm=1.3) / (42 / 2)
            ax.scatter(role_index + offsets, values, s=11, c=color, linewidths=.25,
                       edgecolors="black", zorder=3, gid=f"c_points_{name}_{role}")
            p2 = float(data["benchmarks"][reference][role])
            ax.plot([role_index - .28, role_index + .28], [p2, p2], color="#A6A6A6", lw=.6,
                    zorder=1, gid=f"c_reference_{name}_{role}")
            ax.plot([role_index - .21, role_index + .21], [median, median], color=color, lw=1.2,
                    zorder=4, gid=f"c_median_{name}_{role}")
            x = 134 + (role_index + .5) / 2 * 42
            label(canvas, 154 + 21 * role_index, 123 + offset + (1 - median) * 13 - 3 * role_index,
                  f"{median:.2f}", f"c_number_{name}_{role}", ha="center",
                  color=color if color != V.LIGHT else "black")
            label(canvas, x, 155 + offset, f"{roles[role_index]}\nn = {len(values)}",
                  f"c_role_{name}_{role}", ha="center", linespacing=1.2)
            rows.extend(dict(dataset=name, role=role, molecule=r["molecule"], agreement=r["consistency"],
                             display_x_offset=display_offset) for r, display_offset in zip(eligible, offsets))
            stats.append(dict(dataset=name, role=role, median=median, eligible_molecules=len(values),
                              p2_reference=p2, difference_ci95=ci))
        canvas.plot([135, 138], [140 + offset] * 2, color="#A6A6A6", lw=.6)
        label(canvas, 139, 140 + offset, "P2 reference", "c_reference_legend_" + name, color="#777777")
    return rows, stats


def inventory(conditions):
    width, height = 100., 120.
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), dpi=200)
    ax = fig.add_axes([40 / width, 7 / height, 54 / width, 88 / height])
    ax.set(xlim=(.5, 6.5), ylim=(18.5, .5), xticks=range(1, 7), yticks=range(1, 19))
    ax.set_xticklabels(F.SOLVENTS, rotation=45, ha="left", rotation_mode="anchor")
    formulas = {"Cs2CO3": [("Cs", 7, 0), ("2", 5.25, .55), ("CO", 7, 0), ("3", 5.25, .55)],
                "K2CO3": [("K", 7, 0), ("2", 5.25, .55), ("CO", 7, 0), ("3", 5.25, .55)],
                "KHCO3": [("KHCO", 7, 0), ("3", 5.25, .55)]}
    ax.set_yticklabels(["" if name in formulas else name for name in F.ADDITIVES])
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.scatter(conditions.matrix_row.astype(float), conditions.matrix_column.astype(float), s=15,
               c=V.DARK, linewidths=0, gid="inventory_47_measured_combinations")
    ax.set_gid("SNAr_inventory")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.add_axes([0, 0, 1, 1], zorder=-1)
    canvas.set(xlim=(0, width), ylim=(height, 0))
    canvas.axis("off")
    for name, tokens in formulas.items():
        widths = [renderer.get_text_width_height_descent(text, FontProperties(family="Arial", size=size), False)[0]
                  * 25.4 / fig.dpi for text, size, _ in tokens]
        x = 40 - 3 * PT_MM - sum(widths)
        y = 25 + (F.ADDITIVES.index(name) + .5) / 18 * 88 + .85
        for i, ((text, size, lower), token_width) in enumerate(zip(tokens, widths)):
            canvas.text(x, y + lower, text, fontsize=size, ha="left", va="baseline", gid=f"inventory_{name}_{i}")
            x += token_width
    return fig


def geometry(paths):
    result = []
    for svg in paths:
        path = svg.with_suffix(".cdxml")
        root = ET.parse(path).getroot()
        nodes = {n.get("id"): np.asarray([float(x) for x in n.get("p").split()])
                 for n in root.iter("n") if n.get("p")}
        bonds, adjacency = {}, {k: [] for k in nodes}
        for b in root.iter("b"):
            a, z = b.get("B"), b.get("E")
            if a in nodes and z in nodes:
                adjacency[a].append(z)
                adjacency[z].append(a)
                bonds[frozenset((a, z))] = b.get("Order", "1")
        measured = []
        for atom, neighbors in adjacency.items():
            if len(neighbors) < 2:
                continue
            for i, left in enumerate(neighbors):
                for right in neighbors[:i]:
                    a, b = nodes[left] - nodes[atom], nodes[right] - nodes[atom]
                    angle = float(np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
                    orders = [bonds[frozenset((atom, v))] for v in neighbors]
                    kind = "sp" if "3" in orders else "sp2" if "2" in orders else "unassigned_2D"
                    expected = [180.] if kind == "sp" else [108., 120., 126., 132.] if kind == "sp2" else []
                    error = min(abs(angle - x) for x in expected) if expected else None
                    measured.append(dict(atom=atom, angle_deg=angle, geometry=kind,
                                         nearest_expected_error_deg=error, within_5deg=error <= 5 if expected else None))
        result.append(dict(file=str(path), sha256=sha(path), measured_angles=measured,
                           flagged=[x for x in measured if x["within_5deg"] is False]))
    return result


def rectangle_distance(a, b):
    return float(np.hypot(max(a[0] - b[2], b[0] - a[2], 0), max(a[1] - b[3], b[1] - a[3], 0)))


def readback(fig, layout):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_to_pt = 72 / fig.dpi
    texts = [dict(text=t.get_text(), gid=t.get_gid(), fontsize=t.get_fontsize(),
                  bbox_px=list(t.get_window_extent(renderer).extents))
             for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    assert min(t["fontsize"] for t in texts) >= 5
    point_boxes = []
    for ax in fig.axes:
        for collection in ax.collections:
            if not (collection.get_gid() or "").startswith(("b_points_", "c_points_")):
                continue
            xy = collection.get_offset_transform().transform(collection.get_offsets())
            radius = np.sqrt(collection.get_sizes().max()) / 2 * fig.dpi / 72
            point_boxes.extend((float(x - radius), float(y - radius), float(x + radius), float(y + radius)) for x, y in xy)
    gaps = [(rectangle_distance(t["bbox_px"], p) * px_to_pt, t["text"], t["gid"])
            for t in texts for p in point_boxes]
    minimum = min(gaps)
    assert minimum[0] >= 2, minimum
    line_boxes = []
    for ax in fig.axes[1:]:
        for line in ax.lines:
            box = line.get_window_extent(renderer).extents
            pad = line.get_linewidth() * fig.dpi / 144
            line_boxes.append((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad))
    line_gaps = [(rectangle_distance(t["bbox_px"], box) * px_to_pt, t["text"], t["gid"])
                 for t in texts for box in line_boxes]
    minimum_line = min(line_gaps)
    assert minimum_line[0] >= 2, minimum_line
    width_px, height_px = fig.canvas.get_width_height()
    clipped = [t for t in texts if t["bbox_px"][0] < -.5 or t["bbox_px"][1] < -.5
               or t["bbox_px"][2] > width_px + .5 or t["bbox_px"][3] > height_px + .5]
    assert not clipped, clipped
    structures = [row["visible_bounds_mm"] for row in layout]
    vertical_gap = min(structures[i + 1][1] - structures[i][3] for i in range(len(structures) - 1))
    assert vertical_gap >= 1.0
    reaction_gaps = []
    canvas = fig.axes[0]
    for i, structure in enumerate(structures, 1):
        artists = [t for t in canvas.texts if t.get_gid() == f"a_reaction_{i}"
                   or (t.get_gid() or "").startswith(f"a_reaction_{i}_part_")]
        points = np.concatenate([canvas.transData.inverted().transform(
            artist.get_window_extent(renderer).get_points()) for artist in artists])
        reaction_gaps.append(float(structure[0] - points[:, 0].max()))
    assert min(reaction_gaps) / PT_MM >= 2
    return dict(width_mm=WIDTH, height_mm=HEIGHT, text=texts, minimum_text_to_point_clearance_pt=minimum[0],
                closest_label=dict(text=minimum[1], gid=minimum[2]),
                minimum_text_to_data_line_clearance_pt=minimum_line[0],
                closest_line_label=dict(text=minimum_line[1], gid=minimum_line[2]),
                minimum_structure_vertical_clearance_mm=vertical_gap,
                minimum_reaction_label_to_structure_clearance_mm=min(reaction_gaps), clipped_text_count=len(clipped),
                panel_letters=[t for t in texts if (t["gid"] or "").startswith("panel_")])


def main():
    arrays, conditions, data, arts, inputs, rule_sha = source_data()
    before = {str(path): sha(path) for path in inputs}
    with V.style(), plt.rc_context({"font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT,
                                   "ytick.labelsize": FONT, "svg.hashsalt": "p2-v6-figure2-draft"}):
        fig, canvas = V.new_figure(HEIGHT)
        canvas.set_zorder(10)
        canvas.patch.set_alpha(0)
        layout = breadth(fig, canvas, arrays, arts)
        summaries = summary(fig, canvas, arrays)
        _, external_stats = external(fig, canvas, data)
        c.apply_journal(fig)
        rendered = readback(fig, layout)
        angles = geometry([CHEM / name for name in arrays.structure_file])
        overflow = [row for row in summaries if row["overflow"]]
        assert len(overflow) == 1 and overflow[0]["screen"] == "C-O competitive reactions|1"
        report = dict(schema="p2_v6_figure2", frozen_rule_sha256=rule_sha,
            screen_order=arrays.screen.tolist(), representative_smiles=arrays.smiles.tolist(),
            halide_screens=10, all_screens=15, condition_inventory_count=47,
            master_bond_pt=14.4, master_atom_font_pt=10, uniform_placement_scale=.7,
            geometry=angles, layout=layout, summary_points=summaries, external_stats=external_stats,
            panel_b_display=dict(y_limits_pct=[0,150],ticks_pct=[0,50,100,150],overflow_points=overflow),
            external_result_sha256=RESULT_SHA,external_derivatives_public=False)
        tables=c.tables(2)
        tables["V6_summary_points"]=pd.DataFrame(summaries)
        c.save(fig,"Figure2",report,tables,{"clearance":rendered})
        c.save(inventory(conditions),"FigS28",dict(measured_conditions=47))
        c.write_json("figure2_selection.json",dict(rule_sha256=rule_sha,
            selection=tables["Rule2_selected"].to_dict("records"),
            all_candidates=tables["Rule2_all_candidates"].to_dict("records")))
    assert before == {str(path): sha(path) for path in inputs}, "Read-only input changed"
    return {"panels":{"A":{"screen_count":15,"halide_owner_count":10},
                      "B":{"summary_points":summaries},"C":{"external_result_sha256":RESULT_SHA,"statistics":external_stats}}}


if __name__ == "__main__":
    c.cli(main,2)

"""Merged Figure 1 draft, preserving frozen examples and native ChemDraw scale."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

import main_figure_common as c

HEIGHT = 215.0
TOP = 51.0
ROW_GAP = .9
LAYOUT = {
    "amine": dict(panel="C", x_structure=1.0, structure_width=29.3, x_heat=32.0,
                  heat_width=53.0, strip_top=166.0, tag="Fixed halide"),
    "halide": dict(panel="D", x_structure=92.0, structure_width=29.3, x_heat=123.0,
                   heat_width=54.0, strip_top=181.0, tag="Fixed amine"),
}


def prepare_payload():
    from figure2_data import extract
    payload, _ = extract()
    retained = c.load_payload()
    assert payload["condition_order"] == retained["condition_order"]
    assert payload["selected_second_amine"] == retained["selected_second_amine"]
    assert payload["correlations"] == retained["correlations"]
    for role in ("amine_rows", "halide_rows"):
        for actual, expected in zip(payload[role], retained[role]):
            for field in ("id", "name", "smiles", "pair", "focal_smiles"):
                assert actual[field] == expected[field], (role, field, actual["id"])
            for field in ("pattern_points", "yield_fraction"):
                assert np.array_equal(np.asarray(actual[field], float), np.asarray(expected[field], float), equal_nan=True)
    manifest = json.loads((c.ROOT / "02_FIGURES/ChemDraw/v5/f1/manifest.json").read_text(encoding="utf-8"))
    lookup = {row["id"]: row for row in manifest}
    for role in ("amine", "halide"):
        for row in payload[f"{role}_rows"]:
            record = lookup[row["id"]]
            assert record["smiles"] == row["smiles"]
            path = c.ROOT / record["svg"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == record["svg_sha256"]
            row["display_structure_svg"] = record["svg"]
    return payload, manifest


def best_condition(payload):
    shared = next(row for row in payload["amine_rows"] if row["shared"])
    values = np.asarray(shared["yield_fraction"], dtype=float)
    maximum = float(np.nanmax(values))
    tied = np.flatnonzero(values == maximum).tolist()
    chosen = tied[0]
    metadata = {row["condition_id"]: row for row in payload["condition_metadata"]}
    result = dict(metadata[payload["condition_order"][chosen]])
    result.update(column=chosen + 1, measured_yield_fraction=maximum,
                  measured_yield_pct=maximum * 100,
                  tied_condition_ids=[payload["condition_order"][i] for i in tied],
                  tie_rule="first in frozen condition-only order among exact measured-yield maxima",
                  selection_basis="shared-pair measured yield, not preference",
                  pair=shared["pair"],
                  yield_source=payload["sources"]["measurements"],
                  condition_source=payload["sources"]["conditions"])
    assert maximum * 100 == payload["shared_pair_best_yield_pct"]
    shared_h = next(row for row in payload["halide_rows"] if row["shared"])
    assert shared_h["yield_fraction"] == shared["yield_fraction"]
    return result


def retained_tables():
    tables, mapping = {}, []
    for old in (1, 2):
        for sheet, frame in c.read_tables(old).items():
            name = f"F{old}_{sheet}"
            assert len(name) <= 31 and name not in tables
            tables[name] = frame
            mapping.append(dict(formal_figure=old, formal_sheet=sheet, draft_sheet=name,
                                formal_workbook=f"04_SOURCE_DATA/Figure{old}/Figure{old}_SourceData.xlsx",
                                purpose="retained unchanged formal source table"))
    tables["V5_source_sheet_map"] = pd.DataFrame(mapping)
    return tables


def reaction_and_plate(canvas, payload, best):
    c.panel(canvas, "A", 2, 4, "C–N ligand library")
    c.panel(canvas, "B", 140, 4)
    shared_a = next(row for row in payload["amine_rows"] if row["shared"])
    shared_h = next(row for row in payload["halide_rows"] if row["shared"])
    boxes = []
    for role, row, x, y, color, label in (
        ("halide", shared_h, 2, 18, c.BLUE, "kept in C"),
        ("amine", shared_a, 30, 16, c.ORANGE, "kept in D"),
    ):
        width, height = c.structure(canvas, row["display_structure_svg"], x, y, f"structure_A_{role}")
        rect = Rectangle((x - .6, y - .6), width + 1.2, height + 1.2, fill=False,
                         edgecolor=color, lw=.5, gid=f"A_{role}_box", zorder=6)
        canvas.add_patch(rect)
        canvas.text(x + width / 2, 35, label, ha="center", va="center", gid=f"A_{role}_kept")
        boxes.append(dict(role=role, x=x - .6, y=y - .6, width=width + 1.2, height=height + 1.2))
    canvas.text(26.7, 24, "+", ha="center", va="center", gid="A_plus")
    arrow = FancyArrowPatch((50.5, 25), (90, 25), arrowstyle="->", mutation_scale=8,
                            lw=.7, color="black", gid="A_reaction_arrow", zorder=5)
    canvas.add_patch(arrow)
    formula = lambda value: re.sub(r"(\d+)", lambda match: "$_{" + match.group(1) + "}$", value)
    recipe = f"{formula(best['precatalyst'])}, {best['added_ligand']}\n{formula(best['base'])}, {best['solvent']}"
    canvas.text(69, 13, recipe, ha="center", va="center", linespacing=1.2, gid="A_condition_recipe")
    canvas.text(69, 34, "95 conditions:\nPd 87, Cu 7, Ni 1", ha="center", va="center",
                linespacing=1.2, gid="A_condition_count")
    c.structure(canvas, "02_FIGURES/ChemDraw/Fig1_A_product.svg", 95, 16, "structure_A_product")
    canvas.text(112, 37.5, f"best measured yield {best['measured_yield_pct']:.0f}%",
                ha="center", va="center", gid="A_yield")
    amines, halides = payload["amine_rows"], payload["halide_rows"]
    shared_ai = next(i for i, row in enumerate(amines) if row["shared"])
    shared_hi = next(i for i, row in enumerate(halides) if row["shared"])
    raw = pd.read_csv(c.ROOT / payload["sources"]["measurements"], usecols=["screen", "subpair"])
    measured = set(raw.loc[raw.screen.eq("Buchwald-Hartwig|1"), "subpair"])
    plate = []
    x0, y0, dx, dy = 145, 9, 4.6, 3.35
    for i, halide in enumerate(halides):
        for j, amine in enumerate(amines):
            x, y = x0 + dx * j, y0 + dy * i
            observed = amine["smiles"] + " >> " + halide["smiles"] in measured
            color = c.BLUE if i == shared_hi else c.ORANGE if j == shared_ai else c.LIGHT
            canvas.add_patch(Circle((x, y), .98, facecolor="#E5E5E5" if observed else "white",
                                    edgecolor=color, lw=.6 if i == shared_hi or j == shared_ai else .4,
                                    gid=f"B_well_{i}_{j}", zorder=3))
            if i == shared_hi and j == shared_ai:
                canvas.add_patch(Circle((x, y), .42, facecolor=c.ORANGE, edgecolor="white",
                                        lw=.4, gid="B_shared_pair", zorder=4))
            plate.append(dict(halide=halide["smiles"], amine=amine["smiles"], measured=observed,
                              fixed_halide=i == shared_hi, fixed_amine=j == shared_ai, x_mm=x, y_mm=y))
    selected_y, selected_x = y0 + dy * shared_hi, x0 + dx * shared_ai
    canvas.plot([x0 - 1.7, x0 + dx * 6 + 1.7], [selected_y] * 2,
                lw=.5, color=c.BLUE, gid="B_kept_halide_row", zorder=1)
    canvas.plot([selected_x] * 2, [y0 - 1.7, y0 + dy * 9 + 1.7],
                lw=.5, color=c.ORANGE, gid="B_kept_amine_column", zorder=1)
    canvas.text(x0 + dx * 3, 43, "amines", fontsize=8, ha="center", va="center",
                gid="B_amine_axis")
    canvas.text(139, y0 + dy * 4.5, "aryl halides", fontsize=8, rotation=90,
                ha="center", va="center", gid="B_halide_axis")
    return plate, boxes, recipe


def draw_heatmap(fig, canvas, payload, role, best):
    spec = LAYOUT[role]
    letter = spec["panel"]
    c.panel(canvas, letter, spec["x_structure"], 47, spec["tag"])
    rows = payload[f"{role}_rows"]
    sizes = [c.structure_size(row["display_structure_svg"]) for row in rows]
    heights = np.asarray([height + ROW_GAP for _, height in sizes])
    edges = np.r_[0., np.cumsum(heights)]
    centers = (edges[:-1] + edges[1:]) / 2
    assert max(width for width, _ in sizes) <= spec["structure_width"]
    assert TOP + edges[-1] < spec["strip_top"] - 4
    heat = c.axes_mm(fig, spec["x_heat"], TOP, spec["heat_width"], edges[-1])
    heat.set_gid(f"{letter}_preference_heatmap")
    values = np.asarray([row["pattern_points"] for row in rows], dtype=float)
    assert values.shape == (len(rows), 95)
    shown = heat.pcolormesh(np.arange(96), edges, np.ma.masked_invalid(values),
                            cmap=c.HEAT, vmin=-30, vmax=30, shading="flat", antialiased=False, linewidth=0)
    shown.set_gid(f"{letter}_{role}_actual_preference")
    for i, j in np.argwhere(~np.isfinite(values)):
        heat.add_patch(Rectangle((j, edges[i]), 1, heights[i], facecolor="#E5E5E5",
                                 edgecolor="#777777", hatch="////", linewidth=.15,
                                 gid=f"{letter}_missing_{i}_{j}"))
    # Both markers use the same frozen measured-yield maximum column.
    marker_x = spec["x_heat"] + spec["heat_width"] * (best["column"] - .5) / 95
    canvas.plot([marker_x], [49.4], marker="v", markersize=3, linestyle="None",
                color=c.DARK, markeredgewidth=.5,
                gid=f"{letter}_best_measured_condition_triangle", zorder=4)
    canvas.text(marker_x + 2, 47.5, "best", fontsize=8, ha="left", va="center",
                gid=f"{letter}_best_measured_condition_label")
    heat.set(xlim=(0, 95), ylim=(edges[-1], 0), xticks=[], yticks=[])
    heat.spines[["top", "right", "left", "bottom"]].set_visible(False)
    heat.set_xlabel("95 conditions", labelpad=3)
    heat.xaxis.label.set_gid(f"{letter}_condition_axis")
    alignment, cells = [], []
    for i, (row, (width, height)) in enumerate(zip(rows, sizes)):
        x = spec["x_structure"] + (spec["structure_width"] - width) / 2
        y = TOP + centers[i] - height / 2
        c.structure(canvas, row["display_structure_svg"], x, y, f"structure_{row['id']}")
        alignment.append(dict(panel=letter, role=role, row=i + 1, id=row["id"], name=row["name"],
                              smiles=row["smiles"], pair=row["pair"], structure_svg=row["display_structure_svg"],
                              structure_smiles=row["smiles"], heatmap_row_smiles=row["smiles"],
                              center_y_mm=float(TOP + centers[i]), top_mm=float(TOP + edges[i]),
                              bottom_mm=float(TOP + edges[i + 1]), structure_x_mm=float(x),
                              structure_top_mm=float(y), structure_width_mm=float(width),
                              structure_height_mm=float(height), structure_label_pt=8,
                              bond_length_setting=14.4, condition_column_order="condition metadata only"))
        for j, (condition, value) in enumerate(zip(payload["condition_order"], row["pattern_points"])):
            cells.append(dict(panel=letter, role=role, row=i + 1, id=row["id"], smiles=row["smiles"],
                              pair=row["pair"], column=j + 1, condition_id=condition,
                              preference_points=value, measured=value is not None,
                              best_measured_condition=j + 1 == best["column"], source=payload["sources"]["measurements"]))
    correlations = [dict(record, panel=letter) for record in payload["correlations"] if record["role"] == role]
    assert len(correlations) == len(rows) * (len(rows) - 1) // 2
    mean = float(np.mean([record["r"] for record in correlations]))
    assert mean == payload["mean_correlations"][role]
    strip = c.axes_mm(fig, spec["x_heat"], spec["strip_top"], spec["heat_width"], 5)
    strip.set_gid(f"{letter}_pairwise_r_strip")
    r = np.asarray([record["r"] for record in correlations])
    jitter = .18 * np.sin(np.arange(len(r)) * 2.4)
    strip.scatter(r, jitter, s=9, facecolor="white", edgecolor=c.BLUE if role == "amine" else c.ORANGE,
                  linewidth=.5, gid=f"{letter}_pairwise_r_points", zorder=3)
    strip.plot([mean] * 2, [-.34, .34], color=c.BLUE if role == "amine" else c.ORANGE,
               lw=1, gid=f"{letter}_pairwise_r_mean", zorder=2)
    strip.axvline(0, color="#D0D0D0", lw=.5, gid=f"{letter}_r_zero", zorder=1)
    strip.set(xlim=(-.65, .85), ylim=(-.5, .5), yticks=[], xticks=[-.5, 0, .5])
    strip.spines[["top", "right", "left"]].set_visible(False)
    strip.set_xlabel("Pairwise agreement, r", labelpad=3)
    strip.xaxis.label.set_gid(f"{letter}_r_axis_label")
    canvas.text(spec["x_heat"] + spec["heat_width"], spec["strip_top"] - 3,
                f"mean r {mean:.2f}", ha="right", va="center", gid=f"{letter}_mean_r")
    geometry = dict(role=role, panel=letter, top_mm=TOP, total_height_mm=float(edges[-1]),
                    visible_structure_height_sum_mm=float(sum(height for _, height in sizes)),
                    row_gap_mm=ROW_GAP, heatmap_width_mm=spec["heat_width"],
                    r_strip_top_mm=spec["strip_top"], r_strip_xlim=[-.65, .85],
                    r_points=len(correlations), r_summary="mean of dependent descriptive pairwise correlations; no CI")
    return alignment, cells, correlations, geometry


def agreement(fig, canvas, tables):
    c.panel(canvas, "E", 2, 184, "Molecular agreement")
    rows = tables["F2_C_molecule_agreement"]
    medians = tables["F2_C_group_medians"]
    ax = c.axes_mm(fig, 38, 188, 33, 16)
    ax.set_gid("E_molecular_agreement")
    ax.set(xlim=(-.23, .83), ylim=(3.5, -.5), yticks=[0, 1, 2, 3],
           yticklabels=["Aryl halide", "Amine", "Amine", "Carboxylic acid"], xticks=[0, .5])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    groups = [("Buchwald-Hartwig|1", 2, c.BLUE), ("Buchwald-Hartwig|1", 1, c.ORANGE),
              ("Acylation|1", 1, c.ORANGE), ("Acylation|1", 2, c.DARK)]
    printed = []
    for y, (screen, role, color) in enumerate(groups):
        selected = rows[rows.row_type.eq("molecule") & rows.screen.eq(screen) & rows.role.eq(role)]
        values = selected.sort_values("molecule").agreement.to_numpy(float)
        median = float(medians[medians.screen.eq(screen) & medians.role.eq(role)]["median"].iloc[0])
        ax.scatter(values, y + .16 * np.sin(np.arange(len(values)) * 2.4), s=8,
                   facecolor="white", edgecolor=color, linewidth=.45, zorder=2,
                   gid=f"E_molecule_points_{y}")
        ax.plot([median] * 2, [y - .29, y + .29], color=color, lw=.9,
                gid=f"E_median_line_{y}", zorder=3)
        canvas.text(73, 188 + (y + .5) * 4, f"median {median:.2f}", ha="left", va="center",
                    gid=f"E_median_{y}")
        printed.append(dict(panel="E", quantity=f"{screen} role {role} median", value=median,
                            display=f"median {median:.2f}", source="F2_C_group_medians"))
    ax.set_xlabel("Agreement across partners, r", labelpad=3)
    ax.xaxis.label.set_gid("E_r_axis_label")
    canvas.text(2, 192, "C–N", va="center", gid="E_CN")
    canvas.text(2, 200, "Amide", va="center", gid="E_amide")
    return printed


def render(payload, manifest):
    best = best_condition(payload)
    tables = retained_tables()
    with c.style():
        fig, canvas = c.new_figure(HEIGHT)
        plate, boxes, recipe = reaction_and_plate(canvas, payload, best)
        alignment, cells, correlations, geometry = [], [], [], []
        for role in ("amine", "halide"):
            placed, actual, rs, block = draw_heatmap(fig, canvas, payload, role, best)
            alignment.extend(placed)
            cells.extend(actual)
            correlations.extend(rs)
            geometry.append(block)
        printed = agreement(fig, canvas, tables)
        canvas.text(123, 201, "Preference beyond\nlibrary average (points)", va="center",
                    linespacing=1.2, gid="shared_preference_colorbar_title")
        cbax = c.axes_mm(fig, 123, 206, 54, 2.5)
        cbax.set_gid("shared_preference_colorbar")
        cb = fig.colorbar(ScalarMappable(norm=Normalize(-30, 30), cmap=c.HEAT), cax=cbax,
                          orientation="horizontal", ticks=[-30, 0, 30])
        cb.solids.set_rasterized(False)
        cb.outline.set_linewidth(.5)
        cb.ax.tick_params(width=.5, length=2, pad=2)
        assert len(alignment) == 17 and len(cells) == 17 * 95 and len(correlations) == 66
        tables["V5_row_alignment"] = pd.DataFrame(alignment)
        tables["V5_heatmap_cells"] = pd.DataFrame(cells)
        tables["V5_correlations"] = pd.DataFrame(correlations)
        tables["V5_plate"] = pd.DataFrame(plate)
        tables["V5_best_condition"] = pd.DataFrame([{k: json.dumps(v) if isinstance(v, list) else v for k, v in best.items()}])
        tables["V5_structure_geometry"] = pd.DataFrame([{k: v for k, v in row.items() if k != "trials"} for row in manifest])
        printed += [dict(panel="A", quantity="shared-pair measured maximum yield", value=best["measured_yield_pct"],
                         display=f"best measured yield {best['measured_yield_pct']:.0f}%", source=best["yield_source"])]
        printed += [dict(panel="A", quantity=key, value=value, display=str(value), source="F2_B_condition_metadata")
                    for key, value in (("condition count", 95), ("Pd conditions", 87), ("Cu conditions", 7), ("Ni conditions", 1))]
        printed += [dict(panel=LAYOUT[role]["panel"], quantity="mean pairwise correlation", value=payload["mean_correlations"][role],
                         display=f"mean r {payload['mean_correlations'][role]:.2f}", source=payload["sources"]["correlations"])
                    for role in ("amine", "halide")]
        printed += [dict(panel="C,D", quantity="condition columns", value=95, display="95 conditions", source="F2_B_condition_metadata")]
        printed += [dict(panel="C,D", quantity="correlation tick", value=value, display=f"{value:g}", source="display axis") for value in (-.5, 0, .5)]
        printed += [dict(panel="shared", quantity="preference colorbar tick", value=value, display=f"{value:g}", source="display scale") for value in (-30, 0, 30)]
        tables["V5_printed_numbers"] = pd.DataFrame(printed)
        extra = dict(row_alignment=alignment, block_geometry=geometry, best_condition=best,
                     condition_recipe=recipe, condition_order=payload["condition_order"],
                     condition_order_basis="retained condition-only metadata order, unchanged",
                     rule1_selection=payload["selected_second_amine"], frozen_examples_reexecuted=True,
                     heatmap_range_points=[-30, 30], heatmap_quantity="Preference beyond library average (points)",
                     heatmap_is_raw_yield=False, r_strips_shared_xlim=[-.65, .85],
                     plate_pairs=len(plate), measured_pairs=sum(row["measured"] for row in plate),
                     reaction_boxes=boxes, structure_geometry=manifest, structure_label_pt=8,
                     source_tables="All formal Figure1 and Figure2 tables retained with F1_ and F2_ namespaces",
                     mapping=dict(old1_A="A", old1_B="B", old1_C="E", old2_A="C", old2_B="C point strip",
                                  old2_C="D", old2_D="D point strip"),
                     role_correspondence="blue halide box to blue plate row; orange amine box to orange plate column",
                     best_condition_marker="dark-gray downward triangle above each heatmap; shared measured-yield maximum column",
                     numeric_annotation_exceptions="A library composition and E four group medians explicitly required by v5 workorder")
        return c.save(fig, 1, tables, extra=extra)


def main():
    payload, manifest = prepare_payload()
    c.write_json(1, "figure1_payload.json", payload)
    paths = render(payload, manifest)
    print(json.dumps(dict(paths=paths, best_condition=best_condition(payload),
                          rule1_selection=payload["selected_second_amine"]), ensure_ascii=False))


if __name__ == "__main__":
    c.cli(main, 1)

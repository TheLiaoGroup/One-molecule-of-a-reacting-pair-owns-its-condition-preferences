"""Draft Figure 2: the same comparison across fifteen screens."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.text import Text

import main_figure_common as V
from figure3_selection import calculate, clean, verify_rule


NUMBER = 2
HEIGHT = 215.0
STRUCTURES = V.ROOT / "02_FIGURES/ChemDraw/v3/f3"
SOURCE = "05_DATA/Analysis/figure_display/v5_reference_tables/screen_shares.xlsx"
ORDER = ["Buchwald-Hartwig|1", "Buchwald-Hartwig|2", "C-O competitive reactions|1", "Heck|1", "Hiyama|1",
         "Kumada|1", "SNAr|1", "Sonogashira|1", "Suzuki|1", "Ullmann|1", "Acylation|1", "Chan-Lam|1",
         "Esterification|1", "Grignard|1", "Reductive-amination|1"]
ADDITIVES = ["AcOH", "MsOH", "PTSA", "TFA", "none", "DIPEA", "TEA", "2,6-lutidine", "Py", "DBU",
             "Cs2CO3", "K2CO3", "KHCO3", "KOAc", "KOH", "KOtBu", "NaOMe", "NaOtAm"]
SOLVENTS = ["DMSO", "THF", "t-AmOH", "NMP", "CPME", "MeCN"]
REACTION_LABELS = {
    "Buchwald-Hartwig|1": "C\u2013N (Pd ligand)",
    "Buchwald-Hartwig|2": "C\u2013N (Pd/Cu/Ni)",
    "C-O competitive reactions|1": "C\u2013O (Pd/Cu)",
    "Heck|1": "Heck",
    "Hiyama|1": "Hiyama (Pd)",
    "Kumada|1": "Kumada (Pd/Ni/Fe)",
    "SNAr|1": r"S$_{\mathrm{N}}$Ar",
    "Sonogashira|1": "Sonogashira",
    "Suzuki|1": "Suzuki (Pd)",
    "Ullmann|1": "Ullmann C\u2013O (Cu)",
    "Acylation|1": "Amide coupling",
    "Chan-Lam|1": "Chan\u2013Lam (Cu)",
    "Esterification|1": "Esterification",
    "Grignard|1": "Grignard addition",
    "Reductive-amination|1": "Reductive amination",
}


def _role_colour(role, owner):
    role = str(role).replace("\u2013", "-")
    if "halide" in role:
        return V.BLUE
    if role in ("N-H partner", "N-H nucleophile", "amine"):
        return V.ORANGE
    return V.DARK if owner else V.LIGHT


def _source_tables():
    tables = V.read_tables(3)
    arrays = tables["Fig3A_arrays"].copy()
    assert arrays.screen.tolist() == ORDER
    assert len(arrays) == 15 and int(arrays.halide_array.sum()) == 10
    assert int(arrays.partner_share.lt(0).sum()) == 3
    fresh = clean(calculate())
    assert fresh["rule_sha256"] == verify_rule()
    selected = pd.DataFrame(fresh["selection"])
    retained = tables["Rule2_selected"]
    identity = ["screen", "smiles", "owner_role", "threshold_used", "product_forming_partners"]
    assert selected[identity].to_dict("records") == retained[identity].to_dict("records"), "Frozen Rule 2 selected objects differ"
    assert arrays[["screen", "smiles"]].to_dict("records") == selected[["screen", "smiles"]].to_dict("records")
    conditions = tables["Fig3B_conditions"].copy()
    assert len(conditions) == conditions.condition_id.nunique() == 47
    assert len(conditions[["matrix_row", "matrix_column"]].drop_duplicates()) == 47
    assert set(conditions.base_or_acid) == set(ADDITIVES)
    assert set(conditions["Standardized Name"]) == set(SOLVENTS)
    return tables, arrays, conditions, fresh


def _artist_bounds(canvas, artists, renderer):
    """Return ordered mm bounds after the canvas's inverted-y transformation."""
    points = []
    inverse = canvas.transData.inverted()
    for artist in artists:
        if not isinstance(artist, (Polygon, Text)):
            continue
        box = artist.get_window_extent(renderer)
        corners = inverse.transform(box.get_points())
        padding = artist.get_linewidth() * 25.4 / 144 if isinstance(artist, Polygon) else 0
        points.extend((np.min(corners, axis=0) - padding, np.max(corners, axis=0) + padding))
    assert points
    points = np.asarray(points)
    return (*np.min(points, axis=0), *np.max(points, axis=0))


def _visible_structure_bounds(fig, canvas, paths):
    """Measure native text as well as bonds once, without rescaling a master."""
    groups = []
    for i, path in enumerate(paths):
        before = set(canvas.get_children())
        V.structure(canvas, path, 0, 0, f"measure_owner_{i}")
        groups.append([item for item in canvas.get_children() if item not in before])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = []
    for group in groups:
        bounds.append(_artist_bounds(canvas, group, renderer))
        for item in group:
            item.remove()
    return bounds


def _dot_offsets(values, xlim, width_mm):
    """Separate coincident points vertically; horizontal values remain exact."""
    points = []
    offsets = np.zeros(len(values))
    candidates = (0.0, 1.8, -1.8, 3.6, -3.6, 5.4, -5.4)
    for i in sorted(range(len(values)), key=lambda index: (values[index], index)):
        x_mm = (values[i] - xlim[0]) / (xlim[1] - xlim[0]) * width_mm
        chosen = next((y for y in candidates if all((x_mm - x) ** 2 + (y - other_y) ** 2 >= 1.7 ** 2
                                                   for x, other_y in points)), None)
        assert chosen is not None, "Molecular-share point spacing exceeds the draft strip"
        offsets[i] = chosen
        points.append((x_mm, chosen))
    return offsets


def _draw_arrays(fig, canvas, arrays):
    V.panel(canvas, "A", 1, 3.5)
    paths = [STRUCTURES / row.structure_file for row in arrays.itertuples()]
    bounds = _visible_structure_bounds(fig, canvas, paths)
    heights = np.asarray([box[3] - box[1] + 0.75 for box in bounds])
    edges = 3.2 + np.r_[0, np.cumsum(heights)]
    edges[10:] += 1.2
    centers = []
    for i in range(15):
        top = edges[i]
        bottom = top + heights[i]
        centers.append((top + bottom) / 2)
    assert edges[-1] < 209
    bar_bottom = centers[-1] + 3.0
    bars = V.axes_mm(fig, 79, edges[0], 41, bar_bottom - edges[0])
    bars.set(xlim=(-8, 50), ylim=(bar_bottom, edges[0]), xticks=[0, 20, 40], yticks=[])
    bars.spines[["left", "right", "top"]].set_visible(False)
    bars.axvline(0, color="#D0D0D0", linewidth=0.5, zorder=0)
    canvas.text(99.5, 212.5, "Preference share (%)", fontsize=8, va="center",
                ha="center", gid="A_axis_label")
    divider = edges[10] - 0.6
    canvas.plot([1, 79], [divider, divider], color="black", linewidth=0.5, gid="A_halide_group_divider")
    layout = []
    for i, (row, y, box, path) in enumerate(zip(arrays.itertuples(), centers, bounds, paths), 1):
        label = REACTION_LABELS[row.screen] + ("*" if bool(row.aryl_bromide_only) else "")
        canvas.text(1, y, label, fontsize=8, va="center", ha="left", linespacing=1.12, gid=f"A_reaction_{i}")
        x = 53.5 - (box[0] + box[2]) / 2
        top = y - (box[1] + box[3]) / 2
        width, height = V.structure(canvas, path, x, top, f"A_owner_structure_{i}")
        owner_role = int(row.owner_role)
        for owner, offset, side in ((True, -1.15, owner_role), (False, 1.15, 3 - owner_role)):
            value = 100 * float(getattr(row, f"share{side}"))
            bars.barh(y + offset, value, left=0, height=1.65,
                      color=_role_colour(getattr(row, f"role{side}"), owner), edgecolor="none",
                      gid=f"A_{'owner' if owner else 'partner'}_share_{i}")
        if row.screen == "SNAr|1":
            for side, offset, value in (("owner", -3.6, row.owner_share), ("partner", 3.6, row.partner_share)):
                bars.text(100 * value + 1.7, y + offset, f"{100 * value:.1f}%", fontsize=8,
                          va="center", ha="left", color="black", gid="A_SNAr_" + side + "_number")
        layout.append(dict(screen=row.screen, file=path.relative_to(V.ROOT).as_posix(),
                           center_y_mm=y, structure_x_mm=x, structure_y_mm=top,
                           structure_width_mm=width, structure_height_mm=height,
                           row_top_mm=edges[i - 1], row_bottom_mm=edges[i - 1] + heights[i - 1],
                           atom_font_pt=8, native_subscript_font_pt=6, common_bond_mm=4.064,
                           owner_share_pct=100 * row.owner_share, partner_share_pct=100 * row.partner_share))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    actual_bounds = []
    for i, placement in enumerate(layout, 1):
        artists = [item for item in canvas.get_children()
                   if (item.get_gid() or "").startswith(f"A_owner_structure_{i}|")]
        actual = _artist_bounds(canvas, artists, renderer)
        assert actual[1] >= placement["row_top_mm"] + .1, (i, actual, placement)
        assert actual[3] <= placement["row_bottom_mm"] - .1, (i, actual, placement)
        placement.update(visible_left_mm=actual[0], visible_top_mm=actual[1],
                         visible_right_mm=actual[2], visible_bottom_mm=actual[3])
        actual_bounds.append(actual)
    for i in range(14):
        assert actual_bounds[i + 1][1] - actual_bounds[i][3] >= .3, (i + 1, actual_bounds[i:i + 2])
    return pd.DataFrame(layout)


def _draw_molecular(fig, canvas, arrays):
    V.panel(canvas, "B", 125, 3.5)
    points = V.axes_mm(fig, 129, 11.0, 47, 16)
    points.set(xlim=(0, 65), ylim=(-8, 8), xticks=[0, 25, 50], yticks=[])
    points.spines[["left", "right", "top"]].set_visible(False)
    values = 100 * arrays.share.to_numpy(float)
    offsets = _dot_offsets(values, (0, 65), 47)
    points.scatter(values, offsets, s=18, color=V.DARK, linewidths=0.5,
                   edgecolors="black", gid="B_15_molecular_shares")
    median = float(np.median(values))
    points.plot([median, median], [4.0, 6.5], color="black", linewidth=0.5, gid="B_median_tick")
    points.text(median + 3, 6, f"{median:.0f}%", fontsize=8, va="center", ha="left", gid="B_median_number")
    canvas.text(152.5, 34, "Molecular share (%)", fontsize=8, ha="center", va="center", gid="B_axis_label")
    return pd.DataFrame(dict(screen=arrays.screen, molecular_share=arrays.share,
                             plotted_share_pct=values, presentation_vertical_offset_mm=offsets))


def _draw_snar(fig, canvas, conditions):
    V.panel(canvas, "C", 125, 43, tag=r"S$_{\mathrm{N}}$Ar")
    geometry = []
    for key, center_y in (("halide", 61), ("nucleophile", 84), ("product", 112)):
        path = V.ROOT / "02_FIGURES/ChemDraw" / f"Fig2_B_{key}.svg"
        width, height = V.structure_size(path)
        x, y = 151.5 - width / 2, center_y - height / 2
        V.structure(canvas, path, x, y, "C_reaction_" + key)
        geometry.append(dict(role=key, file=path.relative_to(V.ROOT).as_posix(),
                             x_mm=x, y_mm=y, width_mm=width, height_mm=height,
                             atom_font_pt=8, native_subscript_font_pt=6, common_bond_mm=4.064))
    canvas.text(151.5, 72.5, "+", fontsize=8, ha="center", va="center", gid="C_plus")
    canvas.add_patch(FancyArrowPatch((151.5, 95), (151.5, 101), arrowstyle="->",
                                    mutation_scale=6, linewidth=0.5, color="black", gid="C_reaction_arrow"))
    matrix = V.axes_mm(fig, 147, 137, 30, 69)
    matrix.set(xlim=(0.5, 6.5), ylim=(18.5, 0.5), xticks=range(1, 7), yticks=range(1, 19))
    labels = {"2,6-lutidine": "2,6-lut.", "Cs2CO3": "Cs$_2$CO$_3$", "K2CO3": "K$_2$CO$_3$", "KHCO3": "KHCO$_3$"}
    matrix.set_xticklabels(SOLVENTS, fontsize=8, rotation=90, ha="left", rotation_mode="anchor")
    matrix.set_yticklabels([labels.get(name, name) for name in ADDITIVES], fontsize=8)
    matrix.xaxis.tick_top()
    matrix.tick_params(axis="both", length=0, pad=2)
    for spine in matrix.spines.values():
        spine.set_visible(False)
    matrix.scatter(conditions.matrix_row, conditions.matrix_column, s=13, color=V.DARK,
                   linewidths=0, gid="C_47_measured_combinations")
    return pd.DataFrame(geometry)


def _printed_numbers(arrays):
    rows = []
    def add(panel, quantity, value, display, field, selection):
        rows.append(dict(panel=panel, quantity=quantity, value=float(value), display=display,
                         source_file=SOURCE, source_field=field, source_selection=selection,
                         unit="percent coordinate" if quantity == "axis tick" else "fraction",
                         display_transform="identity" if quantity == "axis tick" else "100 * value"))
    snar = arrays[arrays.screen.eq("SNAr|1")].iloc[0]
    for side in ("owner", "partner"):
        value = float(snar[side + "_share"])
        add("A", "SNAr " + side + " share", value, f"{100 * value:.1f}%", "Fig3A_arrays:" + side + "_share", "screen=SNAr|1; source fraction; display percent")
    median = float(arrays.share.median())
    add("B", "median molecular share", median, f"{100 * median:.0f}%", "Fig3A_arrays:share", "median of 15 source fractions; display percent")
    for panel, ticks in (("A", (0, 20, 40)), ("B", (0, 25, 50))):
        for tick in ticks:
            add(panel, "axis tick", tick, str(tick), "display coordinate", "percentage coordinate scale")
    return pd.DataFrame(rows)


def build():
    tables, arrays, conditions, selection = _source_tables()
    with V.style():
        fig, canvas = V.new_figure(HEIGHT)
        layout = _draw_arrays(fig, canvas, arrays)
        points = _draw_molecular(fig, canvas, arrays)
        reaction = _draw_snar(fig, canvas, conditions)
    draft_tables = dict(tables)
    draft_tables.update(V5_printed_numbers=_printed_numbers(arrays), V5_structure_layout=layout,
                        V5_molecular_points=points, V5_reaction_layout=reaction)
    extra = dict(presentation="formal v5", rule2_selection=selection, frozen_rule2_sha256=selection["rule_sha256"],
                 rule2_selected_objects_unchanged=True, arrays=15, halide_arrays=10,
                 negative_partner_shares=3, snar_measured_cells=47,
                 chemical_atom_font_pt=8, native_chemical_subscript_font_pt=6,
                 chemical_common_bond_mm=4.064,
                 main_figure_source=SOURCE,
                 removed_v3_elements=["variance-composition bars", "row backgrounds", "table rules", "sentence titles", "median-owner summary"])
    return fig, draft_tables, extra


def main():
    fig, tables, extra = build()
    V.write_json(2, "figure2_selection.json", extra["rule2_selection"])
    with V.style():
        return V.save(fig, NUMBER, tables, extra)


if __name__ == "__main__":
    V.cli(main, 2)

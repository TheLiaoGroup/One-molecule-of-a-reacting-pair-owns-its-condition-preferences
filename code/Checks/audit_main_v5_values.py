"""Four-figure ownership adapters around unchanged independent reference math.

No drawing module is imported. Observed tables and artists are observations;
reference calculations remain in the existing independent numerical modules.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from numbers import Real
from pathlib import Path
import json
import re
import tempfile

import numpy as np
import openpyxl
import pandas as pd

import audit_main_v2_values23 as values23
import audit_main_v3_values12 as values12
import audit_main_v3_values35 as values35
import audit_main_v3_values4 as values4

TOL = values12.TOL
SEMANTIC = {1: ("overview", "preference"), 2: ("screen_shares",),
            3: ("molecule_context",), 4: ("transfer",)}
OLD = {1: 2, 2: 3, 3: 4, 4: 5}
REFERENCE_NAMES = {1: "overview", 2: "preference", 3: "screen_shares", 4: "molecule_context", 5: "transfer"}
ALIASES = {
    "04_SOURCE_DATA/Figure2/figure2_payload.json": (1, "figure1_payload.json"),
    "source_data/Figure2/figure2_payload.json": (1, "figure1_payload.json"),
    "04_SOURCE_DATA/Figure3/figure3_selection.json": (2, "figure2_selection.json"),
    "source_data/Figure3/figure3_selection.json": (2, "figure2_selection.json"),
    "04_SOURCE_DATA/Figure5/figure5_selection.json": (4, "figure4_selection.json"),
    "source_data/Figure5/figure5_selection.json": (4, "figure4_selection.json"),
}


def _find(root, *names):
    paths = [Path(root) / name for name in names]
    found = [path for path in paths if path.is_file()]
    assert found, "Missing v5 independent input: " + " or ".join(map(str, paths))
    return found[0]


def source_ownership(value, reverse=False):
    if not isinstance(value, str):
        return value
    replacements = [(f"04_SOURCE_DATA/Figure{number}/Figure{number}_SourceData.xlsx",
                     f"05_DATA/Analysis/figure_display/v5_reference_tables/{semantic}.xlsx")
                    for number, semantic in REFERENCE_NAMES.items()]
    replacements += [("04_SOURCE_DATA/Figure2/figure2_payload.json",
                      "05_DATA/Analysis/figure_display/v5_reference_tables/preference_payload.json"),
                     ("04_SOURCE_DATA/Figure5/", "04_SOURCE_DATA/Figure4/"),
                     ("06_CODE/Figures/figure5.py", "06_CODE/Figures/figure4.py")]
    for old, new in reversed(replacements) if reverse else replacements:
        value = value.replace(new, old) if reverse else value.replace(old, new)
    return value


@contextmanager
def ownership_paths(artifact_dir=None):
    """Redirect renamed evidence ownership without changing a calculation."""
    originals = {module: module._path for module in (values12, values35)}
    for module, original in originals.items():
        def adapted(root, *names, _original=original):
            for name in names:
                if name in ALIASES:
                    number, filename = ALIASES[name]
                    if artifact_dir is not None:
                        sidecar = Path(artifact_dir) / filename
                        assert sidecar.is_file(), "Missing explicit artifact sidecar: " + str(sidecar)
                        return sidecar
                    candidates = [Path(root) / f"04_SOURCE_DATA/Figure{number}/{filename}",
                                  Path(root) / f"source_data/Figure{number}/{filename}"]
                    present = [path for path in candidates if path.is_file()]
                    assert present, "Missing renamed v5 evidence: " + filename
                    return present[0]
            return _original(root, *names)
        module._path = adapted
    try:
        yield
    finally:
        for module, original in originals.items():
            module._path = original


@contextmanager
def directional_public_reader(number):
    """Use public measurement fields for the serial v5 directional check."""
    if number != 3:
        yield
        return
    original_directional, original_raw = values4._directional, values23._raw
    values4._directional = values23._directional.__wrapped__
    values23._raw = values35._raw
    try:
        yield
    finally:
        values4._directional, values23._raw = original_directional, original_raw


def legacy_tables(number, tables):
    if number == 1:
        result = {name[3:]: frame.copy(deep=True) for name, frame in tables.items() if name.startswith("F2_")}
        molecules = result["C_molecule_agreement"]
        medians = result["C_group_medians"]
        assert set(molecules.row_type) == {"molecule"}, "Unexpected split molecule row types"
        assert len(medians) == 4 and not medians.duplicated(["screen", "role"]).any(), \
            "Split group medians must have four unique screen-role keys"
        # The retained workbook separates summary rows; the legacy numerical
        # interface expects a combined view of these same observed values.
        summary = medians.rename(columns={"median": "agreement"}).copy(deep=True)
        summary["row_type"] = "median of the group"
        result["C_molecule_agreement"] = pd.concat([molecules, summary], ignore_index=True)
    else:
        result = {name: frame.copy(deep=True) for name, frame in tables.items()}
        assert "Retained_v3_numbers" in result, "Missing retained numerical ledger"
        result["Printed_numbers"] = result["Retained_v3_numbers"].copy(deep=True)
    for frame in result.values():
        for column in frame:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(lambda value: source_ownership(value, reverse=True))
    return result


def read_stored_workbook(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        tables = {}
        for sheet in book:
            rows = list(sheet.iter_rows(values_only=True))
            while rows and all(value is None for value in rows[-1]):
                rows.pop()
            if not rows:
                tables[sheet.title] = pd.DataFrame()
                continue
            data = [[np.nan if value is None or value == "" else value for value in row]
                    for row in rows[1:]]
            tables[sheet.title] = pd.DataFrame(data, columns=rows[0], dtype=object)
        return tables
    finally:
        book.close()


def retained_cell_equal(actual, expected):
    if pd.isna(actual) or pd.isna(expected):
        return bool(pd.isna(actual) and pd.isna(expected))
    if isinstance(actual, str) or isinstance(expected, str):
        return isinstance(actual, str) and isinstance(expected, str) and actual == expected
    if isinstance(actual, (bool, np.bool_)) or isinstance(expected, (bool, np.bool_)):
        return (isinstance(actual, (bool, np.bool_)) and isinstance(expected, (bool, np.bool_))
                and actual == expected)
    if isinstance(actual, Real) and isinstance(expected, Real):
        return actual == expected
    return type(actual) is type(expected) and actual == expected


def retained_cells(number, root, tables):
    cells = 0
    for semantic in SEMANTIC[number]:
        path = _find(root, *[f"{prefix}/v5_reference_tables/{semantic}.xlsx" for prefix in
                           ("05_DATA/Analysis/figure_display", "results/figure_display", "figure_inputs")])
        reference = read_stored_workbook(path)
        for sheet, expected in reference.items():
            name = (("F1_" if semantic == "overview" else "F2_") + sheet) if number == 1 else (
                "Retained_v3_numbers" if sheet == "Printed_numbers" else sheet)
            assert name in tables, "Retained source sheet missing: " + name
            actual = tables[name]
            assert actual.columns.tolist() == expected.columns.tolist() and actual.shape == expected.shape, \
                "Retained source shape differs: " + name
            for column in expected:
                expected[column] = expected[column].map(source_ownership)
            a, b = actual.to_numpy(object), expected.to_numpy(object)
            assert all(retained_cell_equal(x, y) for x, y in zip(a.flat, b.flat)), \
                "Retained source cell differs: " + name
            cells += int(a.size)
    return dict(exact_retained_cells=cells, reference_workbooks=list(SEMANTIC[number]))


def audit_tables(number, root, tables, artifact_dir=None):
    assert number in OLD
    exact = retained_cells(number, root, tables)
    observed = legacy_tables(number, tables)
    with ownership_paths(artifact_dir), directional_public_reader(number):
        module = values12 if number == 1 else values4 if number == 3 else values35
        independent = module.audit_tables(OLD[number], Path(root), observed)
    return dict(**exact, independent_calculation=independent, tolerance=TOL,
                legacy_calculations_unchanged=True)


def _items(readback, kind):
    rows = [row for axis in readback["axes"] for row in axis.get(kind, []) if row.get("gid")]
    names = [row["gid"] for row in rows]
    assert len(names) == len(set(names)), "Duplicate quantitative artist identifiers"
    return {row["gid"]: row for row in rows}


def _close(actual, expected, label):
    return values12.Evidence().close("v5 " + label, actual, expected)


def _text(readback, gid):
    rows = [row for row in readback["texts"] if row.get("gid") == gid]
    assert len(rows) == 1, "Missing unique actual text: " + gid
    return rows[0]["text"]


def _plain(text):
    text = re.sub(r"\$_\{?(\d+)\}?\$", r"\1", str(text))
    return " ".join(text.split())


def _structure_in_row(rb, prefix, top, bottom):
    labels = [item for item in rb["texts"] if (item.get("gid") or "").startswith(prefix + "|label")]
    assert labels, "v5 actual native structure labels missing: " + prefix
    centers = [rb["height_mm"] * (1 - (item["bbox"][1] + item["bbox"][3] / 2) / rb["height_px"])
               for item in labels]
    assert all(top <= center <= bottom for center in centers), "v5 actual structure-row alignment: " + prefix


@lru_cache(maxsize=4)
def _preference(root):
    evidence = values12.Evidence()
    return values12._preference_reference(Path(root), evidence)


MEASUREMENTS = "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz"
CORRELATIONS = "06_CODE/Figures/preference_data.py:measurement_tables/load_data:Fig1BC_r"


def _canonical_expected(number, root):
    """Frozen display inventory with values from independent numerical references."""
    root = str(Path(root).resolve())
    rows = []
    if number == 1:
        ref = _preference(root)
        def add(panel, quantity, value, display, source):
            rows.append(dict(panel=panel, quantity=quantity, value=value, display=display, source=source))
        for screen, role in ((values12.CN, 2), (values12.CN, 1), (values12.AMIDE, 1), (values12.AMIDE, 2)):
            median = float(ref["agreements"].query("screen == @screen and role == @role").agreement.median())
            add("E", f"{screen} role {role} median", median, f"median {median:.2f}", "F2_C_group_medians")
        add("A", "shared-pair measured maximum yield", ref["best"], f"best measured yield {ref['best']:.0f}%", MEASUREMENTS)
        count = len(ref["metadata"])
        add("A", "condition count", count, str(count), "F2_B_condition_metadata")
        for metal in ("Pd", "Cu", "Ni"):
            value = int(ref["metadata"].metal.eq(metal).sum())
            add("A", metal + " conditions", value, str(value), "F2_B_condition_metadata")
        for role, panel in (("amine", "C"), ("halide", "D")):
            mean = float(ref["correlations"].query("role == @role").r.mean())
            add(panel, "mean pairwise correlation", mean, f"mean r {mean:.2f}", CORRELATIONS)
        add("C,D", "condition columns", count, f"{count} conditions", "F2_B_condition_metadata")
        for value in (-.5, 0, .5):
            add("C,D", "correlation tick", value, f"{value:g}", "display axis")
        for value in (-30, 0, 30):
            add("shared", "preference colorbar tick", value, f"{value:g}", "display scale")
    elif number == 2:
        arrays, _ = values35._anatomy(root)
        source = "05_DATA/Analysis/figure_display/v5_reference_tables/screen_shares.xlsx"
        def add(panel, quantity, value, display, field, selection):
            tick = quantity == "axis tick"
            rows.append(dict(panel=panel, quantity=quantity, value=float(value), display=display,
                             source_file=source, source_field=field, source_selection=selection,
                             unit="percent coordinate" if tick else "fraction",
                             display_transform="identity" if tick else "100 * value"))
        snar = arrays.set_index("screen").loc["SNAr|1"]
        for side in ("owner", "partner"):
            value = float(snar[side + "_share"])
            add("A", "SNAr " + side + " share", value, f"{100 * value:.1f}%",
                "Fig3A_arrays:" + side + "_share", "screen=SNAr|1; source fraction; display percent")
        median = float(arrays.share.median())
        add("B", "median molecular share", median, f"{100 * median:.0f}%", "Fig3A_arrays:share",
            "median of 15 source fractions; display percent")
        for panel, ticks in (("A", (0, 20, 40)), ("B", (0, 25, 50))):
            for value in ticks:
                add(panel, "axis tick", value, str(value), "display coordinate", "percentage coordinate scale")
    elif number == 3:
        rows = [dict(panel=panel, quantity="axis tick", value=value,
                     display=f"{value:.1f}" if value else "0.0", source_file="display coordinate",
                     source_field="axis tick", source_selection="condition-preference fraction")
                for panel, ticks in (("A", (-.4, 0., .4)), ("B", (-.4, 0., .4, .8))) for value in ticks]
    elif number == 4:
        data = values35._transfer(root)
        common = dict(aggregation="100 * (1 - sum(SSE)/sum(SST))")
        for cohort in ("Set A", "Set B"):
            pooled = data["a"].query("set == @cohort and row_type == 'pooled'")
            assert len(pooled) == 1, "v5 canonical pooled reference inventory"
            row = pooled.iloc[0]
            for side, field in (("owner", "sse_e"), ("partner", "sse_n")):
                value = 100 * float(row[side])
                rows.append(dict(panel="A", value=value, display=f"{value:.1f}",
                                 source_file="05_DATA/Results/" + values35.LOSS_FILES[cohort],
                                 source_field="sst;" + field, source_selection="all rows", **common,
                                 cohort=cohort, side=side, row_type="pooled", history_partners=None))
        for side in ("owner", "partner"):
            endpoint = data["pooled"].query("side == @side and partners == 6")
            assert len(endpoint) == 1, "v5 canonical history endpoint reference inventory"
            value = float(endpoint.iloc[0].explained_pct)
            rows.append(dict(panel="C", value=value, display=f"{value:.1f}%",
                             source_file="05_DATA/Results/" + values35.LOSS_FILES["depth"], source_field="sst;sse_6",
                             source_selection="common_target=True;side=" + side, **common,
                             cohort=None, side=side, row_type=None, history_partners=6))
    else:
        raise AssertionError("Unknown v5 canonical figure")
    return pd.DataFrame(rows)


def _record_table(actual, expected, label):
    assert actual.columns.tolist() == expected.columns.tolist(), f"v5 {label} column inventory"
    assert len(actual) == len(expected), f"v5 {label} row inventory"
    for i, reference in enumerate(expected.to_dict("records")):
        for column, value in reference.items():
            observed = actual.iloc[i][column]
            name = f"{label} row {i + 1} {column}"
            if pd.isna(value):
                assert pd.isna(observed), "v5 " + name + " missing-value identity"
            elif isinstance(value, (bool, np.bool_)):
                assert isinstance(observed, (bool, np.bool_)) and observed == value, "v5 " + name + " boolean type/value"
            elif isinstance(value, Real):
                assert isinstance(observed, Real) and not isinstance(observed, (bool, np.bool_)), "v5 " + name + " numeric type"
                _close([observed], [value], name)
            else:
                assert isinstance(observed, str) and observed == value, "v5 " + name + " text/display identity"


def _numeric_text(text):
    text = str(text).replace("\N{MINUS SIGN}", "-").strip()
    return float(text) if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text) else None


def _axis_ticks(rb, axis, ticks, direction, label):
    """Locate saved tick artists by projected position, including axes without GIDs."""
    x, y, width, height = axis["bbox"]
    lo, hi = axis["xlim" if direction == "x" else "ylim"]
    reach = 30 * rb["dpi"] / 72
    found = []
    for item in rb["texts"]:
        value = _numeric_text(item["text"])
        if value is None:
            continue
        tx, ty, tw, th = item["bbox"]
        if direction == "x":
            projected = x + (value - lo) / (hi - lo) * width
            match = abs(tx + tw / 2 - projected) <= 1 and y - reach <= ty <= y and ty + th <= y + 1
        else:
            projected = y + (value - lo) / (hi - lo) * height
            match = ty <= projected <= ty + th and x - reach <= tx <= x and tx + tw <= x + 1
        if match:
            found.append(value)
    _close(sorted(found), sorted(ticks), label + " actual numeric tick labels")


def _axis_with(rb, kind, gid):
    axes = [axis for axis in rb["axes"] if any(item.get("gid") == gid for item in axis.get(kind, []))]
    assert len(axes) == 1, "v5 canonical missing unique artist axis: " + gid
    return axes[0]


def _canonical_artists(number, expected, rb):
    rows = expected.to_dict("records")
    if number == 1:
        for i in range(4):
            assert _text(rb, f"E_median_{i}") == rows[i]["display"], "v5 canonical Figure1 median label"
        assert _text(rb, "A_yield") == rows[4]["display"], "v5 canonical Figure1 yield label"
        counts = [int(row["value"]) for row in rows[5:9]]
        assert _text(rb, "A_condition_count") == f"{counts[0]} conditions:\nPd {counts[1]}, Cu {counts[2]}, Ni {counts[3]}", "v5 canonical Figure1 composition label"
        for i, panel in enumerate(("C", "D")):
            assert _text(rb, panel + "_mean_r") == rows[9 + i]["display"], "v5 canonical Figure1 correlation label"
            assert _text(rb, panel + "_condition_axis") == rows[11]["display"], "v5 canonical Figure1 condition-axis label"
            axes = [axis for axis in rb["axes"] if axis.get("gid") == panel + "_pairwise_r_strip"]
            assert len(axes) == 1, "v5 canonical Figure1 correlation axis"
            _axis_ticks(rb, axes[0], [row["value"] for row in rows[12:15]], "x", "Figure1 " + panel)
        axes = [axis for axis in rb["axes"] if axis.get("gid") == "shared_preference_colorbar"]
        assert len(axes) == 1, "v5 canonical Figure1 colorbar axis"
        _axis_ticks(rb, axes[0], [row["value"] for row in rows[15:18]], "x", "Figure1 colorbar")
    elif number == 2:
        for row, gid in zip(rows[:3], ("A_SNAr_owner_number", "A_SNAr_partner_number", "B_median_number")):
            assert _text(rb, gid) == row["display"], "v5 canonical Figure2 actual percentage label: " + gid
        for gid, kind, ticks, panel in (("A_owner_share_1", "patches", [0, 20, 40], "A"),
                                        ("B_15_molecular_shares", "collections", [0, 25, 50], "B")):
            _axis_ticks(rb, _axis_with(rb, kind, gid), ticks, "x", "Figure2 " + panel)
    elif number == 3:
        for name in values4.DISPLAY_ORDER[::2]:
            _axis_ticks(rb, _axis_with(rb, "lines", "A|path|" + name), [-.4, 0, .4], "y", "Figure3 A " + name)
        _axis_ticks(rb, _axis_with(rb, "lines", "B|owner|1"), [-.4, 0, .4, .8], "x", "Figure3 B")
    else:
        lines = _items(rb, "lines")
        for i, cohort in enumerate(("Set A", "Set B")):
            owner, partner = rows[2 * i:2 * i + 2]
            assert _text(rb, "A|summary|" + cohort) == f"{cohort}\n{owner['display']} / {partner['display']}", "v5 canonical Figure4 pooled label"
            _close(lines["A|pooled|" + cohort]["x"], [partner["value"]], "canonical Figure4 partner point")
            _close(lines["A|pooled|" + cohort]["y"], [owner["value"]], "canonical Figure4 owner point")
        for row in rows[4:]:
            side = row["side"]
            assert _text(rb, "C|summary|" + side) == side.title() + " " + row["display"], "v5 canonical Figure4 endpoint label"
            line = lines["C|pooled|" + side]
            indices = [i for i, value in enumerate(line["x"]) if value == 6]
            assert len(indices) == 1, "v5 canonical Figure4 endpoint artist inventory"
            _close([line["y"][indices[0]]], [row["value"]], "canonical Figure4 endpoint artist")


def _figure1_cells(root, tables, rb):
    assert "V5_heatmap_cells" in tables, "v5 missing Figure1 V5_heatmap_cells"
    ref = _preference(str(Path(root).resolve()))
    shared = next(row for row in ref["partners"]["amine"] if row["shared"])
    best = int(np.nanargmax(np.asarray(shared["yield_fraction"], float))) + 1
    rows = []
    meshes = _items(rb, "meshes")
    for role, panel in (("amine", "C"), ("halide", "D")):
        partners = ref["partners"][role]
        expected_mesh = np.asarray([row["pattern_points"] for row in partners], float)
        _close(meshes[f"{panel}_{role}_actual_preference"]["array"], expected_mesh, "Figure1 cell ledger actual mesh")
        for i, row in enumerate(partners, 1):
            for j, (condition, value) in enumerate(zip(ref["order"], row["pattern_points"]), 1):
                rows.append(dict(panel=panel, role=role, row=i, id=row["id"], smiles=row["smiles"], pair=row["pair"],
                                 column=j, condition_id=condition, preference_points=value, measured=bool(np.isfinite(value)),
                                 best_measured_condition=j == best, source=MEASUREMENTS))
    _record_table(tables["V5_heatmap_cells"], pd.DataFrame(rows), "Figure1 heatmap cell ledger")
    return len(rows)


def audit_current_tables(number, root, tables, readback):
    assert "Printed_numbers" in tables, "v5 missing canonical Printed_numbers"
    expected = _canonical_expected(number, root)
    _record_table(tables["Printed_numbers"], expected, "canonical Printed_numbers")
    _canonical_artists(number, expected, readback)
    cells = _figure1_cells(root, tables, readback) if number == 1 else 0
    return dict(canonical_printed_rows=len(expected), independently_verified_heatmap_cells=cells,
                row_inventory_types_metadata_display_and_actual_artists=True, tolerance=TOL)


def _figure1_artists(root, tables, rb):
    ref = _preference(str(Path(root).resolve()))
    lines, collections = _items(rb, "lines"), _items(rb, "collections")
    meshes = _items(rb, "meshes")
    assert not ({"B_halide_role_leader", "B_amine_role_leader"} & set(lines)), "v5 Figure1 long role connectors"
    for gid, label in (("B_amine_axis", "amines"), ("B_halide_axis", "aryl halides")):
        assert _text(rb, gid) == label, "v5 Figure1 plate label"
        assert next(t for t in rb["texts"] if t.get("gid") == gid)["fontsize"] == 8
    shared = next(row for row in ref["partners"]["amine"] if row["shared"])
    index = int(np.nanargmax(np.asarray(shared["yield_fraction"], float)))
    condition = ref["metadata"].iloc[index]
    best = rb["best_condition"]
    assert best["condition_id"] == ref["order"][index] and best["column"] == index + 1, "v5 Figure1 best condition identity"
    expected_recipe = f"{condition.precatalyst}, {condition.added_ligand} {condition.base}, {condition.solvent}"
    assert _plain(_text(rb, "A_condition_recipe")) == _plain(expected_recipe), "v5 Figure1 recipe identity"
    assert _text(rb, "A_yield") == f"best measured yield {ref['best']:.0f}%", "v5 Figure1 actual yield label"
    alignment = tables["V5_row_alignment"]
    for role, letter in (("amine", "C"), ("halide", "D")):
        partners = ref["partners"][role]
        expected = np.asarray([row["pattern_points"] for row in partners], float)
        mesh = meshes[f"{letter}_{role}_actual_preference"]
        _close(mesh["array"], expected, "Figure1 actual heatmap values")
        axis = next(axis for axis in rb["axes"] if axis.get("gid") == f"{letter}_preference_heatmap")
        assert not axis["lines"], "v5 Figure1 heatmap condition line remains"
        marker = lines[f"{letter}_best_measured_condition_triangle"]
        assert marker["marker"] == "v" and marker["color"].lower() == "#4d4d4d", "v5 Figure1 triangle style"
        assert marker["linestyle"] == "None" and marker["markersize"] == 3, "v5 Figure1 triangle style"
        layout_x, width = (32., 53.) if role == "amine" else (123., 54.)
        _close(marker["x"], [layout_x + width * (index + .5) / 95], "Figure1 triangle column")
        assert marker["y"][0] < rb["block_geometry"][0 if role == "amine" else 1]["top_mm"], "v5 Figure1 triangle above heatmap"
        assert _text(rb, f"{letter}_best_measured_condition_label") == "best", "v5 Figure1 best label"
        rows = alignment[alignment.role.eq(role)].sort_values("row")
        assert rows.id.tolist() == [row["id"] for row in partners], "v5 Figure1 structure-row identities"
        assert rows.smiles.tolist() == rows.structure_smiles.tolist() == rows.heatmap_row_smiles.tolist() == [row["smiles"] for row in partners], "v5 Figure1 structure-row identities"
        coordinates = np.asarray(mesh["coordinates"], float)
        edges = coordinates[:, 0, 1]
        _close(coordinates[:, :, 0], np.tile(np.arange(96), (len(partners) + 1, 1)), "Figure1 actual condition columns")
        bbox, ylim = axis["bbox"], axis["ylim"]
        display_y = bbox[1] + (edges - ylim[0]) / (ylim[1] - ylim[0]) * bbox[3]
        edge_mm = rb["height_mm"] * (1 - display_y / rb["height_px"])
        _close(rows.top_mm, edge_mm[:-1], "Figure1 actual heatmap-row top")
        _close(rows.bottom_mm, edge_mm[1:], "Figure1 actual heatmap-row bottom")
        for row in rows.itertuples():
            _structure_in_row(rb, "structure_" + row.id, row.top_mm, row.bottom_mm)
        correlations = ref["correlations"].query("role == @role").r.to_numpy(float)
        _close(np.asarray(collections[f"{letter}_pairwise_r_points"]["offsets"])[:, 0], correlations, "Figure1 actual pairwise r")
        mean = float(np.mean(correlations))
        _close(lines[f"{letter}_pairwise_r_mean"]["x"], [mean, mean], "Figure1 mean rather than median")
        assert _text(rb, f"{letter}_mean_r") == f"mean r {mean:.2f}", "v5 Figure1 actual mean label"
    for y, (screen, role) in enumerate(((values12.CN, 2), (values12.CN, 1), (values12.AMIDE, 1), (values12.AMIDE, 2))):
        group = ref["agreements"].query("screen == @screen and role == @role").sort_values("molecule")
        _close(np.asarray(collections[f"E_molecule_points_{y}"]["offsets"])[:, 0], group.agreement, "Figure1 actual molecular agreements")
        median = float(np.median(group.agreement))
        _close(lines[f"E_median_line_{y}"]["x"], [median, median], "Figure1 group median rather than mean")
        assert _text(rb, f"E_median_{y}") == f"median {median:.2f}", "v5 Figure1 actual median label"
    return dict(heatmap_cells=1615, row_identities=17, pairwise_correlations=66,
                best_condition=best["condition_id"], best_column=index + 1, recipe_matches=True,
                actual_triangles=2, mean_and_median_distinguished=True)


def _figure2_artists(root, tables, rb):
    lines, patches, collections = (_items(rb, kind) for kind in ("lines", "patches", "collections"))
    arrays = tables["Fig3A_arrays"]
    geometry = tables["V5_structure_layout"]
    chosen, _ = values35._owners(str(Path(root).resolve()))
    values35._frame(tables["Rule2_selected"], chosen, ["screen", "smiles"],
                    ["owner_role", "threshold_used", "product_forming_partners"],
                    label="v5 Figure2 frozen Rule2 object selection")
    assert geometry.screen.tolist() == arrays.screen.tolist(), "v5 Figure2 structure row selection"
    for i, row in enumerate(arrays.itertuples(), 1):
        layout = geometry.iloc[i - 1]
        for side in ("owner", "partner"):
            patch = patches[f"A_{side}_share_{i}"]
            _close([patch["width"]], [100 * getattr(row, side + "_share")], "Figure2 actual signed share")
            offset = -1.15 if side == "owner" else 1.15
            _close([patch["xy"][1] + patch["height"] / 2], [layout.center_y_mm + offset], "Figure2 actual share-row alignment")
        selected = tables["Rule2_selected"].iloc[i - 1]
        assert row.screen == selected.screen and row.smiles == selected.smiles, "v5 Figure2 selected structure identity"
        _structure_in_row(rb, f"A_owner_structure_{i}", layout.row_top_mm, layout.row_bottom_mm)
    values = 100 * arrays.share.to_numpy(float)
    _close(np.asarray(collections["B_15_molecular_shares"]["offsets"])[:, 0], values, "Figure2 actual 15 shares")
    median = float(np.median(values))
    _close(lines["B_median_tick"]["x"], [median, median], "Figure2 median rather than mean")
    assert _text(rb, "B_median_number") == f"{median:.0f}%", "v5 Figure2 actual median label"
    conditions = tables["Fig3B_conditions"]
    _close(collections["C_47_measured_combinations"]["offsets"], conditions[["matrix_row", "matrix_column"]], "Figure2 actual measured combinations")
    return dict(signed_share_bars=30, molecular_share_points=15, measured_combinations=47,
                selected_structures=15, median_not_mean=True)


def _figure3_artists(root, tables, rb):
    lines = _items(rb, "lines")
    assert rb["frozen_amine_order"] == list(values4.DISPLAY_ORDER), "v5 Figure3 frozen amine selection"
    for row in tables["A_eight_amines"].itertuples():
        _close(lines[f"A|path|{row.plot_label}"]["y"], [row.T_amide, row.T_BH1, row.T_BH2], "Figure3 actual context path")
        _close(lines[f"A|path|{row.plot_label}"]["x"], [0, 1, 2], "Figure3 context order")
    for row in tables["B_23_molecules"].itertuples():
        pos = row.plot_row
        for label, value in (("nonowner", row.T_N), ("owner", row.T_O)):
            _close(lines[f"B|{label}|{pos}"]["x"], [value], "Figure3 actual context point")
            _close(lines[f"B|{label}|{pos}"]["y"], [pos], "Figure3 actual row alignment")
        _close(lines[f"B|connector|{pos}"]["x"], [row.T_N, row.T_O], "Figure3 actual connector")
    return dict(context_paths=8, context_points=46, row_identities=23)


def _figure4_artists(root, tables, rb):
    lines = _items(rb, "lines")
    for row in tables["A_array_and_pooled"].itertuples():
        gid = "A|pooled|" + row.set if row.row_type == "pooled" else f"A|{row.set}|{row.screen}"
        _close(lines[gid]["x"], [row.partner_display_pct], "Figure4 actual partner prediction")
        _close(lines[gid]["y"], [row.owner_display_pct], "Figure4 actual owner prediction")
    for pos, row in enumerate(tables["B_equal_depth"].itertuples(), 1):
        for side in ("partner", "owner"):
            _close(lines[f"B|{side}|{row.set}"]["x"], [getattr(row, side + "_display_pct")], "Figure4 actual equal-depth value")
            _close(lines[f"B|{side}|{row.set}"]["y"], [pos], "Figure4 separate equal-depth cohorts")
    for sheet, prefix, grouping in (("C_pooled_curves", "C|pooled|", ["side"]),
                                    ("C_selected_array_curves", "C|thin|", ["screen", "side"])):
        for key, group in tables[sheet].groupby(grouping, sort=False):
            key = key if isinstance(key, tuple) else (key,)
            gid = prefix + "|".join(key)
            group = group.sort_values("partners")
            _close(lines[gid]["x"], group.partners, "Figure4 actual history depths")
            _close(lines[gid]["y"], group.explained_pct, "Figure4 actual history curve")
    assert rb["selected_arrays"] == ["Acylation|1", "Suzuki|1"] and rb["common_targets"] == 310, "v5 Figure4 Rule3 selections"
    return dict(prediction_points=len(tables["A_array_and_pooled"]), equal_depth_cohorts=3,
                selected_screens=["Acylation|1", "Suzuki|1"], common_targets=310)


def audit_artists(number, root, tables, readback):
    return {1: _figure1_artists, 2: _figure2_artists, 3: _figure3_artists, 4: _figure4_artists}[number](root, tables, readback)


def canonical_fault_tests(number, root, tables, readback):
    """Use the same canonical contract for formal and synthetic fault controls."""
    tests = []
    def reject(name, target, fragment, action):
        try:
            action()
        except AssertionError as error:
            assert fragment in str(error), f"Fault {name} rejected for unrelated reason: {error}"
            tests.append(dict(fault=name, target=target, rejected=True, new_issue=str(error), baseline="PASS"))
            return
        raise AssertionError("Planted canonical fault accepted: " + name)
    for name, field in (("wrong canonical value", "value"), ("wrong canonical display", "display")):
        bad = {key: frame.copy(deep=True) for key, frame in tables.items()}
        index = bad["Printed_numbers"].index[0]
        bad["Printed_numbers"].loc[index, field] = float(bad["Printed_numbers"].loc[index, field]) + .1 if field == "value" else "wrong display"
        reject(name, "Printed_numbers:" + field, "canonical Printed_numbers row 1 " + field,
               lambda t=bad: audit_current_tables(number, root, t, readback))
    bad = {key: frame.copy(deep=True) for key, frame in tables.items()}
    bad.pop("Printed_numbers")
    reject("missing canonical ledger", "Printed_numbers", "missing canonical Printed_numbers",
           lambda: audit_current_tables(number, root, bad, readback))
    if number == 1:
        bad = {key: frame.copy(deep=True) for key, frame in tables.items()}
        frame = bad["V5_heatmap_cells"]
        index = frame.index[frame.preference_points.notna()][0]
        frame.loc[index, "preference_points"] += .1
        reject("wrong canonical heatmap cell", "V5_heatmap_cells:preference_points", "Figure1 heatmap cell ledger",
               lambda: audit_current_tables(number, root, bad, readback))
    required = {1: (values12, "04_SOURCE_DATA/Figure2/figure2_payload.json"),
                2: (values35, "04_SOURCE_DATA/Figure3/figure3_selection.json"),
                4: (values35, "04_SOURCE_DATA/Figure5/figure5_selection.json")}
    if number in required:
        module, alias = required[number]
        with tempfile.TemporaryDirectory(prefix="v5_missing_sidecar_") as folder:
            def missing_sidecar():
                with ownership_paths(folder):
                    module._path(Path(root), alias)
            reject("explicit artifact missing own sidecar", ALIASES[alias][1], "Missing explicit artifact sidecar", missing_sidecar)
    return tests


def fault_tests(number, root, tables, readback):
    """Caller must establish the unmodified numerical and artist baseline first."""
    tests = canonical_fault_tests(number, root, tables, readback)
    def reject(name, target, fragment, action):
        try:
            action()
        except AssertionError as error:
            assert fragment in str(error), f"Fault {name} rejected for unrelated reason: {error}"
            tests.append(dict(fault=name, target=target, rejected=True, new_issue=str(error), baseline="PASS"))
            return
        raise AssertionError("Planted fault accepted: " + name)
    bad = {name: frame.copy(deep=True) for name, frame in tables.items()}
    sheet, field = {1: ("F2_C_group_medians", "median"), 2: ("Fig3A_arrays", "owner_share"),
                    3: ("A_eight_amines", "T_amide"), 4: ("A_array_and_pooled", "owner")}[number]
    bad[sheet].loc[bad[sheet].index[0], field] += .1
    reject("wrong source value", f"{sheet}:{field}", "Retained source cell differs", lambda: retained_cells(number, root, bad))
    bad_rb = deepcopy(readback)
    gid, key, value_index = {1: ("C_pairwise_r_mean", "x", 0), 2: ("B_median_tick", "x", 0),
                           3: ("A|path|M4", "y", 0), 4: ("C|pooled|owner", "y", 0)}[number]
    _items(bad_rb, "lines")[gid][key][value_index] += .1
    expected = {1: "Figure1 mean rather than median", 2: "Figure2 median rather than mean",
                3: "Figure3 actual context path", 4: "Figure4 actual history curve"}[number]
    reject("wrong actual plotted value", gid, expected, lambda: audit_artists(number, root, tables, bad_rb))
    if number == 1:
        bad_rb = deepcopy(readback)
        _items(bad_rb, "lines")["C_best_measured_condition_triangle"]["x"][0] += 1
        reject("wrong best condition column", "C_best_measured_condition_triangle", "Figure1 triangle column", lambda: audit_artists(1, root, tables, bad_rb))
        bad_rb = deepcopy(readback)
        next(t for t in bad_rb["texts"] if t.get("gid") == "A_condition_recipe")["text"] = "wrong recipe"
        reject("wrong actual recipe", "A_condition_recipe", "v5 Figure1 recipe identity", lambda: audit_artists(1, root, tables, bad_rb))
    else:
        bad_rb = deepcopy(readback)
        if number == 2:
            selected = {name: frame.copy(deep=True) for name, frame in tables.items()}
            selected["Rule2_selected"].loc[selected["Rule2_selected"].index[0], "smiles"] = "C"
            reject("wrong frozen selected molecule", "Rule2_selected:smiles", "v5 Figure2 frozen Rule2 object selection",
                   lambda: audit_artists(2, root, selected, readback))
            _items(bad_rb, "collections")["C_47_measured_combinations"]["offsets"][0][0] += 1
            fragment = "Figure2 actual measured combinations"
        elif number == 3:
            bad_rb["frozen_amine_order"][0], bad_rb["frozen_amine_order"][1] = bad_rb["frozen_amine_order"][1], bad_rb["frozen_amine_order"][0]
            fragment = "v5 Figure3 frozen amine selection"
        else:
            bad_rb["selected_arrays"] = ["Heck|1", "Suzuki|1"]
            fragment = "v5 Figure4 Rule3 selections"
        reject("wrong selected row or object", "actual selection/row", fragment, lambda: audit_artists(number, root, tables, bad_rb))
    return tests

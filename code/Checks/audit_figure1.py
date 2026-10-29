"""Audit formal Figure 1 against saved display records, without raw-data analysis."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from xml.etree import ElementTree as ET

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from openpyxl import load_workbook
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "06_CODE/Figures"))
from structure_art import load as load_art

INPUT = ROOT / "05_DATA/Analysis/figure_display/v6_figure1"
RECORD_SHA = "1b97e9c409e1f6344dbe2be8c653664542a397b843c4d42bc8c6ee5f3f04eda6"
PORTABLE_RECORD_SHA = "9bf9a633f784eaba08e468b161951476d3152a28235f986b15ad4e8936e0bdfa"
INPUTS_SHA = "0838238d88bfd0e792b7fe0a462a35b499179c9cbf4778c399521afcb6848905"
PORTABLE_INPUTS_SHA = "1aa61e9d05d8b78107f56a03a9c02abb10e38359a7c2ff5f7f66ab1b3a72f0e6"
RULE_SHA = "50e5a7d1003d5d795cdf034f244d25a8324230a5ee00d3d6479763fce3c226c9"
PAIR = "Cc1cc(C)nc(N)c1 >> N#Cc1ccnc(Cl)c1"
NS = {"s": "http://www.w3.org/2000/svg"}
PT = 72 / 25.4
DIMENSIONS_MM = [180, 99]
BAND_X, BAND_WIDTH, BAND_HEIGHT = 34, 112, 3.5
STEP_HEIGHT, SCALE_WIDTH, SCALE_HEIGHT = 16, 28, 1.8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def equal_saved(actual, expected):
    if isinstance(expected, float) and not math.isfinite(expected):
        assert actual is None or not math.isfinite(actual)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            equal_saved(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            equal_saved(a, b)
    else:
        assert actual == expected, (actual, expected)


def node(svg, id):
    nodes = svg.findall(f".//*[@id='{id}']")
    assert len(nodes) == 1, id
    return nodes[0]


def distance(a, b):
    return math.hypot(max(a[0] - b[2], b[0] - a[2], 0), max(a[1] - b[3], b[1] - a[3], 0))


def segment_box(a, b, box):
    a, b = np.asarray(a), np.asarray(b)
    delta = b - a
    low, high = 0., 1.
    for axis in range(2):
        if abs(delta[axis]) < 1e-12:
            if not box[axis] <= a[axis] <= box[axis + 2]:
                low, high = 1., 0.
                break
        else:
            crosses = sorted(((box[axis] - a[axis]) / delta[axis], (box[axis + 2] - a[axis]) / delta[axis]))
            low, high = max(low, crosses[0]), min(high, crosses[1])
    if low <= high:
        return 0.
    corners = np.asarray([[box[x], box[y]] for x in (0, 2) for y in (1, 3)])
    params = np.clip((corners - a) @ delta / (delta @ delta), 0, 1)
    return min(float(np.linalg.norm(corners - (a + params[:, None] * delta), axis=1).min()),
               distance([*a, *a], box), distance([*b, *b], box))


def verify_metadata(meta, saved):
    assert meta["source_records_sha256"] == RECORD_SHA
    assert meta["rule_sha256"] == RULE_SHA
    assert meta["scientific_values_recalculated"] is False
    assert meta["target_pair"] == meta["reference_pair"] == PAIR
    assert meta["condition_permutation"] == saved["condition_permutation"]
    assert meta["sorted_condition_order"] == saved["sorted_condition_order"]
    assert meta["record_summaries"] == [{key: row[key] for key in ("key", "n_pairs", "r", "n_common")}
                                         for row in saved["records"]]
    assert meta["dimensions_mm"] == DIMENSIONS_MM
    assert meta["step_axis_limit_pp"] == 13 and meta["colour_range_pp"] == [-30, 30]
    assert meta["panels"] == [] and meta["panel_titles"] == {}


def fingerprint_geometry(svg):
    strips = {}
    for key in ("reference", "halide_record", "amine_record"):
        band = node(svg, "fingerprint_" + key)
        assert float(band.get("data-width-mm")) == BAND_WIDTH
        assert float(band.get("data-height-mm")) == BAND_HEIGHT
        cells = band.findall("s:rect", NS)
        assert len(cells) == 95
        top = float(cells[0].get("y"))
        for i, cell in enumerate(cells):
            assert abs(float(cell.get("x")) - BAND_X - i * BAND_WIDTH / 95) < 1e-9, (key, i, "x")
            assert float(cell.get("y")) == top and float(cell.get("height")) == BAND_HEIGHT, (key, i, "y/height")
            assert abs(float(cell.get("width")) - (BAND_WIDTH / 95 + .002)) < 1e-9, (key, i, "width")
        strips[key] = [BAND_X, top, BAND_X + BAND_WIDTH + .002, top + BAND_HEIGHT]
    assert 30 <= strips["reference"][1] < strips["halide_record"][1] < strips["amine_record"][1]
    return strips


def annotation_contract(svg):
    labels = {}
    for item in svg.findall(".//s:text", NS):
        role = item.get("data-role")
        assert role in ("annotation", "axis", "atom", "citation", "scheme"), (item.get("id"), role)
        if role == "scheme":
            assert item.get("id") in ("reaction", "plus")
        if role != "annotation" or not re.search(r"[A-Za-z0-9]", item.text or ""):
            continue
        identifier = item.get("data-logical-label") or item.get("id")
        assert identifier
        labels.setdefault(identifier, []).append(item.text or "")
    assert len(labels) <= 12, ("narrative annotation count", len(labels))
    for identifier, parts in labels.items():
        text = " ".join(parts)
        words = re.findall(r"[A-Za-z0-9]+(?:['\u2019][A-Za-z0-9]+)*(?:[.-][A-Za-z0-9]+)*", text)
        limit = 6 if identifier == "question" and text == "New pair: which record carries over?" else 5
        assert len(words) <= limit, ("annotation word count", identifier, text, len(words))
    assert node(svg, "question").text == "New pair: which record carries over?"
    assert node(svg, "measured_label").text == "new pair, measured"
    ordering = (node(svg, "retrospective_order").text or "").lower()
    assert "measured" in ordering and "preference" in ordering
    assert re.search(r"\b(order|sort)", ordering) and "predict" not in ordering
    assert node(svg, "amine_class").text == "ligand guides: amine class"
    obsolete = ("informer_libraries", "robustness_generality", "outcome_models", "scope_measurements",
                "scope_screens", "partner_replacement", "full_experiment_r", "owner_result", "transfer_result")
    for identifier in obsolete:
        assert not svg.findall(f".//*[@id='{identifier}']"), ("obsolete layout element", identifier)
    visible = " ".join(item.text or "" for item in svg.findall(".//s:text", NS)).lower()
    assert not re.search(r"\bp1\b", visible), "P1 is an internal identifier, not a visible label"
    assert "//" not in visible, "Missing measurements must use boxed diagonal cells"
    for text in ("177,039", "0.27 vs 0.07", "10/10", "3\u00d7", "informer libraries", "robustness", "outcome and condition models"):
        assert text not in visible, ("obsolete layout text", text)
    return dict(logical_labels=len(labels), ordinary_maximum_words=5,
                exact_author_requested_question_words=6,
                question_exception_authority="09_SUBMISSION/CODEX_PROMPT_20261028_v6_final_touch.md",
                retrospective_measured_order_visible=True, visible_internal_pair_identifier_absent=True,
                removed_findings_chain_absent=True)


def preference_scale_geometry(svg):
    scale = node(svg, "05_preference_colour_scale")
    cells = scale.findall("s:rect", NS)
    assert len(cells) == 120
    x, y = float(cells[0].get("x")), float(cells[0].get("y"))
    heat = LinearSegmentedColormap.from_list("audit_scale", ["#5A5A5A", "#FFFFFF", "#D9B748"])
    for i, cell in enumerate(cells):
        assert abs(float(cell.get("x")) - x - i * SCALE_WIDTH / 120) < 1e-9
        assert float(cell.get("y")) == y and float(cell.get("height")) == SCALE_HEIGHT
        assert abs(float(cell.get("width")) - (SCALE_WIDTH / 120 + .002)) < 1e-9
        rgba = heat(i / 119)
        assert cell.get("fill") == "#" + "".join(f"{round(value * 255):02x}" for value in rgba[:3])
    bar = [x, y, x + SCALE_WIDTH + .002, y + SCALE_HEIGHT]
    assert x >= BAND_X + BAND_WIDTH
    assert bar[2] <= DIMENSIONS_MM[0] and bar[3] <= DIMENSIONS_MM[1]
    strips = fingerprint_geometry(svg)
    assert bar[3] <= strips["reference"][1], "Shared scale must sit above the first data row"
    gaps = {key: distance(bar, box) * PT for key, box in strips.items()}
    assert gaps["reference"] <= 21.6, ("Preference scale to first measured fingerprint", gaps)
    assert [node(svg, "color_tick_" + value).text for value in ("\u221230", "0", "+30")] == ["\u221230", "0", "+30"]
    ticks = [node(svg, "color_tick_" + value) for value in ("\u221230", "0", "+30")]
    for tick, xx, anchor in zip(ticks, (x, x + SCALE_WIDTH / 2, x + SCALE_WIDTH), ("start", "middle", "end")):
        assert abs(float(tick.get("x")) - xx) < 1e-9 and tick.get("text-anchor") == anchor
    assert len({tick.get("y") for tick in ticks}) == 1
    title = node(svg, "preference_scale")
    assert title.text == "Preference (pp)" and title.get("text-anchor") == "middle"
    assert abs(float(title.get("x")) - x - SCALE_WIDTH / 2) < 1e-9
    assert float(title.get("y")) < float(ticks[0].get("y")) < y
    return dict(colourbar_bbox_mm=bar, strip_bboxes_mm=strips, strip_gaps_pt=gaps,
                measured_strip_gap_pt=gaps["reference"], nearest_explained_strip_gap_pt=min(gaps.values()),
                required_max_pt=21.6, required_explained_strip="reference",
                scale_width_mm=SCALE_WIDTH, scale_height_mm=SCALE_HEIGHT,
                scale_colours_and_dimensions_preserved=True)


def verify_svg(svg, saved):
    assert svg.get("width") == "180mm" and svg.get("height") == "99mm"
    assert not svg.findall(".//s:image", NS)
    ids = [item.get("id") for item in svg.iter() if item.get("id")]
    assert len(ids) == len(set(ids))
    meta = json.loads(svg.find("s:metadata", NS).text)
    verify_metadata(meta, saved)
    strips = fingerprint_geometry(svg)
    for item in meta["primitives"]:
        if item["id"]:
            actual = node(svg, item["id"])
            points = [[float(value) for value in pair.split(",")] for pair in actual.get("points").split()]
            assert points == item["points"] and float(actual.get("stroke-width")) == item["width_mm"]
    heat = LinearSegmentedColormap.from_list("audit_preference", ["#5A5A5A", "#FFFFFF", "#D9B748"])
    cells_count = 0
    missing_count = 0
    for key, values in [("reference", saved["reference"]["pattern_points"])] + [
            (row["key"], row["pattern_points"]) for row in saved["records"]]:
        band = node(svg, "fingerprint_" + key)
        assert float(band.get("data-width-mm")) == BAND_WIDTH
        cells = band.findall("s:rect", NS)
        assert len(cells) == 95
        missing_art = band.findall("s:polyline[@data-role='missing']", NS)
        expected_missing = [i for i, original in enumerate(saved["condition_permutation"])
                            if not math.isfinite(values[original])]
        assert [item.get("data-cell-index") for item in missing_art] == [str(i) for i in expected_missing]
        assert [item.get("data-condition") for item in cells] == saved["sorted_condition_order"]
        for i, (cell, original) in enumerate(zip(cells, saved["condition_permutation"])):
            value = values[original]
            assert int(cell.get("data-original-index")) == original
            assert cell.get("data-value") == (repr(value) if math.isfinite(value) else "missing")
            if math.isfinite(value):
                rgba = heat((min(max(value, -30), 30) + 30) / 60)
                fill = "#" + "".join(f"{round(channel * 255):02x}" for channel in rgba[:3])
            else:
                fill = "#FFFFFF"
                assert cell.get("stroke") == "#B8B8B8" and float(cell.get("stroke-width")) == .10
                hatch = node(svg, f"missing_{key}_{i}")
                x, y = float(cell.get("x")), float(cell.get("y"))
                points = [[float(value) for value in pair.split(",")] for pair in hatch.get("points").split()]
                assert np.allclose(points, [[x, y + BAND_HEIGHT], [x + BAND_WIDTH / 95, y]], rtol=0, atol=1e-9)
                assert hatch.get("stroke") == "#999999" and float(hatch.get("stroke-width")) == .10
                missing_count += 1
            assert cell.get("fill") == fill
        cells_count += len(cells)
    assert len(svg.findall(".//*[@data-role='missing']")) == missing_count
    for key, identifier, hue in (("halide_record", "halide_route", "#0072B2"),
                                 ("amine_record", "amine_route", "#D55E00")):
        route = node(svg, identifier)
        points = [[float(value) for value in pair.split(",")] for pair in route.get("points").split()]
        assert 2 <= len(points) <= 3 and points[0][1] < points[-1][1]
        assert abs(points[-1][1] - (strips[key][1] + BAND_HEIGHT / 2)) < 1e-9
        assert points[-1][0] < BAND_X if key == "halide_record" else points[-1][0] > BAND_X + BAND_WIDTH
        assert route.get("stroke") == hue
    structures = svg.findall(".//*[@data-master]")
    assert len(structures) == 2 and {item.get("id") for item in structures} == {"target_halide", "target_amine"}
    for record in saved["records"]:
        group = node(svg, "steps_" + record["key"])
        assert float(group.get("data-ymin")) == -13 and float(group.get("data-ymax")) == 13
        axis = group.findall("s:polyline", NS)
        points = [[[float(value) for value in pair.split(",")] for pair in item.get("points").split()] for item in axis]
        x, y = points[0][0]
        assert x == BAND_X
        assert float(group.get("data-axis-height-mm")) == STEP_HEIGHT
        assert float(group.get("data-width-mm")) == BAND_WIDTH
        assert points[0] == [[x, y], [x, y + STEP_HEIGHT]]
        assert points[1] == [[x, y + STEP_HEIGHT / 2], [x + BAND_WIDTH, y + STEP_HEIGHT / 2]]
        assert y > strips[record["key"]][3] and y + STEP_HEIGHT <= DIMENSIONS_MM[1]
        assert float(axis[1].get("stroke-width")) == .10
        steps = [item for item in group.findall("s:line", NS) if item.get("data-segment") is not None]
        assert len(steps) == 10
        for i, (item, segment) in enumerate(zip(steps, record["segments"])):
            assert int(item.get("data-segment")) == i
            assert float(item.get("data-mean")) == segment["mean"]
            yy = y + (13 - segment["mean"]) / 26 * STEP_HEIGHT
            assert float(item.get("y1")) == float(item.get("y2")) == yy
            assert float(item.get("x1")) == x + segment["positions"][0] / 95 * BAND_WIDTH
            assert float(item.get("x2")) == x + (segment["positions"][-1] + 1) / 95 * BAND_WIDTH
        assert node(svg, record["key"] + "_correlation").text == (
            f"r = {record['r']:.2f}".replace("-", "\u2212"))
        partners = "amines" if record["key"] == "halide_record" else "halides"
        assert node(svg, record["key"] + "_partners").text == f"{record['n_pairs']} other {partners}"
    for item in svg.findall(".//s:text", NS):
        size = float(item.get("font-size")) * PT
        assert item.get("font-family") == "Arial" and size >= 5
        allowed = (5.25, 7) if item.get("data-role") == "atom" else (5, 7)
        assert round(size, 6) in allowed, (item.get("id"), size)
    question = node(svg, "question")
    assert float(question.get("font-size")) * PT == 7 and question.get("font-weight") == "bold"
    return dict(display_cells=cells_count, saved_steps=20, missing_cells=missing_count,
                missing_cell_convention="white boxed diagonal", metadata=meta,
                annotations=annotation_contract(svg), preference_scale=preference_scale_geometry(svg))


def native_geometry(svg):
    results = []
    masters = {"target_halide": "02_FIGURES/ChemDraw/v5/f1/h2.svg",
               "target_amine": "02_FIGURES/ChemDraw/v5/f1/a2.svg"}
    groups = svg.findall(".//*[@data-master]")
    assert len(groups) == 2 and {item.get("id") for item in groups} == set(masters)
    for id, master in masters.items():
        group = node(svg, id)
        assert group.get("data-master") == master
        source = ROOT / group.get("data-master")
        assert group.get("data-master-sha256") == sha(source)
        art = load_art(source)
        scale = 7 / art["label_px"] / PT
        assert float(group.get("data-native-uniform-scale")) == scale
        polygons = group.findall("s:polygon", NS)
        assert len(polygons) == len(art["polygons"])
        actual = np.asarray([[float(value) for value in pair.split(",")] for pair in polygons[0].get("points").split()])
        shift = actual[0] - art["polygons"][0][0] * scale
        for polygon, expected in zip(polygons, art["polygons"]):
            actual = np.asarray([[float(value) for value in pair.split(",")] for pair in polygon.get("points").split()])
            assert np.allclose(actual, expected * scale + shift, rtol=0, atol=1e-8)
        labels = group.findall("s:text", NS)
        assert len(labels) == len(art["labels"])
        for label, expected in zip(labels, art["labels"]):
            assert label.text == expected["text"]
            assert abs(float(label.get("font-size")) * PT - 7 * expected["scale"]) < 1e-9
            assert abs(float(label.get("x")) - (expected["x"] * scale + shift[0])) < 1e-8
            assert abs(float(label.get("y")) - (expected["y"] * scale + shift[1])) < 1e-8
        cdxml = ET.parse(source.with_suffix(".cdxml")).getroot()
        assert float(cdxml.get("BondLength")) == 14.4
        atoms = {item.get("id"): np.asarray(list(map(float, item.get("p").split())))
                 for item in cdxml.iter("n") if item.get("p")}
        deviations = []
        for bond in cdxml.iter("b"):
            vector = atoms[bond.get("E")] - atoms[bond.get("B")]
            angle = math.degrees(math.atan2(vector[1], vector[0]))
            deviations.append(abs(angle / 15 - round(angle / 15)) * 15)
        assert max(deviations) < .1
        results.append(dict(id=id, native_uniform_geometry=True, master_bond_pt=14.4,
                            max_15degree_grid_deviation_deg=max(deviations)))
    return results


def workbook_check(path, saved):
    book = load_workbook(path, data_only=True, read_only=True)
    rows = list(book["Display_cells"].values)[1:]
    assert len(rows) == 285
    position = 0
    for key, values in [("reference", saved["reference"]["pattern_points"])] + [
            (row["key"], row["pattern_points"]) for row in saved["records"]]:
        for i, original in enumerate(saved["condition_permutation"]):
            row = rows[position]
            position += 1
            assert row[:4] == (key, i, saved["sorted_condition_order"][i], original)
            value = values[original]
            assert row[6] == repr(value)
            assert (row[4] is None) if not math.isfinite(value) else abs(row[4] - value) < 1e-12
    rows = list(book["Saved_steps"].values)[1:]
    assert len(rows) == 20
    for row, (record, i, segment) in zip(rows, [(record, i, segment) for record in saved["records"]
                                               for i, segment in enumerate(record["segments"])]):
        assert row[:3] == (record["key"], i, ",".join(map(str, segment["positions"])))
        assert row[5] == repr(segment["mean"]) and abs(row[3] - segment["mean"]) < 1e-12
    contributors = list(book["Record_contributors"].values)[1:]
    expected_contributors = [(record["key"], i, saved["sorted_condition_order"][i], record["contributor_counts"][original])
                             for record in saved["records"] for i, original in enumerate(saved["condition_permutation"])]
    assert contributors == expected_contributors and len(contributors) == 190
    members = list(book["Member_pairs"].values)[1:]
    expected_members = [(record["key"], id, pair, pair != PAIR) for record in saved["records"]
                        for id, pair in zip(record["member_ids"], record["member_pairs"])]
    assert members == expected_members and len(members) == 15
    printed = {row[0]: row[1] for row in list(book["Printed_numbers"].values)[1:]}
    assert abs(printed["halide record r"] - saved["records"][0]["r"]) < 1e-15
    assert abs(printed["amine record r"] - saved["records"][1]["r"]) < 1e-15
    assert printed["halide full experiment mean pairwise r"] == .27
    assert printed["amine full experiment mean pairwise r"] == .07
    assert [printed[key] for key in ("step lower pp", "step upper pp", "colour lower pp", "colour upper pp")] == [-13, 13, -30, 30]
    book.close()


def layout_check(svg, pdf, meta):
    with pymupdf.open(pdf) as document:
        assert len(document) == 1
        page = document[0]
        assert abs(page.rect.width / PT - DIMENSIONS_MM[0]) < 1e-4 and abs(page.rect.height / PT - DIMENSIONS_MM[1]) < 1e-4
        spans = [span for block in page.get_text("dict")["blocks"] for line in block.get("lines", []) for span in line["spans"]]
        assert spans and all("Arial" in span["font"] for span in spans)
        assert all(span["size"] >= 4.999 for span in spans)
        assert all(page.rect.contains(pymupdf.Rect(span["bbox"])) for span in spans)
        used_fonts = [font for font in page.get_fonts() if font[2] == "TrueType"]
        assert len(used_fonts) == 2
        assert all(len(document.extract_font(font[0])[3]) > 1000 for font in used_fonts)
        actual = {}
        for label in meta["texts"]:
            matches = [span for span in spans if span["text"] == label["text"]]
            assert matches, label["text"]
            expected = label["bbox_mm"]
            span = min(matches, key=lambda item: abs(item["bbox"][0] / PT - expected[0]) + abs(item["bbox"][1] / PT - expected[1]))
            actual[label["id"]] = np.asarray(span["bbox"]) / PT
    critical = [row for row in meta["texts"] if row["role"] == "annotation"]
    text_gaps = [(distance(actual[a["id"]], actual[b["id"]]) * PT, a["id"], b["id"])
                 for i, a in enumerate(critical) for b in critical[i + 1:]]
    assert min(text_gaps)[0] >= 2, min(text_gaps)
    segments = [item for key in ("halide_record", "amine_record")
                for item in node(svg, "steps_" + key).findall("s:line", NS)]
    step_gaps = [(max(0, segment_box([float(item.get("x1")), float(item.get("y1"))],
                                    [float(item.get("x2")), float(item.get("y2"))], actual[row["id"]])
                          - float(item.get("stroke-width")) / 2) * PT, row["id"])
                 for row in critical for item in segments]
    assert min(step_gaps)[0] >= 2, min(step_gaps)
    critical_routes = [item for item in meta["primitives"] if item["role"] in ("route", "literature", "separator")]
    route_gaps = [(max(0, segment_box(a, b, actual[row["id"]]) - item["width_mm"] / 2) * PT,
                   row["id"], item["id"]) for row in critical
                  for item in critical_routes for a, b in zip(item["points"], item["points"][1:])]
    assert min(route_gaps)[0] >= 2, min(route_gaps)
    for id in ("halide_route", "amine_route"):
        assert len(node(svg, id).get("points").split()) <= 3
    scale = preference_scale_geometry(svg)
    strip_route_gaps = [(max(0, segment_box(a, b, box) - item["width_mm"] / 2) * PT, key, item["id"])
                        for key, box in scale["strip_bboxes_mm"].items() for item in critical_routes
                        for a, b in zip(item["points"], item["points"][1:])]
    assert min(strip_route_gaps)[0] >= 2, min(strip_route_gaps)
    bar = scale["colourbar_bbox_mm"]
    scale_ids = [item.get("id") for item in node(svg, "05_preference_colour_scale").findall("s:text", NS)]
    scale_text_gaps = [(distance(actual[left], box) * PT, left, right)
                       for left in scale_ids for right, box in actual.items() if left != right]
    bar_text_gaps = [(distance(bar, box) * PT, name) for name, box in actual.items()]
    scale_route_gaps = [(max(0, segment_box(a, b, box) - item["width_mm"] / 2) * PT, name, item["id"])
                        for name, box in [("colourbar", bar)] + [(name, actual[name]) for name in scale_ids]
                        for item in critical_routes for a, b in zip(item["points"], item["points"][1:])]
    assert min(scale_text_gaps)[0] >= 2, min(scale_text_gaps)
    assert min(bar_text_gaps)[0] >= 2, min(bar_text_gaps)
    assert min(scale_route_gaps)[0] >= 2, min(scale_route_gaps)
    return dict(embedded_Arial=True, text_in_page=True, min_annotation_gap_pt=min(text_gaps),
                min_annotation_step_gap_pt=min(step_gaps), min_strip_route_gap_pt=min(strip_route_gaps),
                min_annotation_route_gap_pt=min(route_gaps),
                preference_scale=scale, min_scale_text_gap_pt=min(scale_text_gaps),
                min_colourbar_text_gap_pt=min(bar_text_gaps), min_scale_route_gap_pt=min(scale_route_gaps))


def negative_controls(svg, saved):
    results = []
    def reject(name, mutate, check=None):
        clone = copy.deepcopy(svg)
        mutate(clone)
        try:
            (check(clone) if check else verify_svg(clone, saved))
        except AssertionError:
            results.append(dict(test=name, rejected=True))
        else:
            raise AssertionError("negative control accepted: " + name)
    def change_meta(clone, key, value):
        item = clone.find("s:metadata", NS)
        meta = json.loads(item.text)
        meta[key] = value
        item.text = json.dumps(meta)
    reject("changed_saved_r", lambda clone: change_meta(clone, "record_summaries", []))
    reject("changed_visible_partner_count", lambda clone: setattr(node(clone, "halide_record_partners"), "text", "7 other amines"))
    reject("changed_condition_order", lambda clone: node(clone, "fingerprint_reference")[0].set("data-condition", "C0"))
    reject("changed_cell_value", lambda clone: node(clone, "fingerprint_reference")[0].set("data-value", "999"))
    reject("changed_step_scale", lambda clone: node(clone, "steps_halide_record").set("data-ymax", "12"))
    reject("changed_step_mean", lambda clone: node(clone, "steps_amine_record").find("s:line", NS).set("data-mean", "999"))
    reject("changed_P1_identity", lambda clone: change_meta(clone, "target_pair", "wrong"))
    reject("undersized_font", lambda clone: node(clone, "measured_label").set("font-size", "1"))
    reject("changed_source_hash", lambda clone: change_meta(clone, "source_records_sha256", "0" * 64))
    def old_scale_position(clone):
        for cell in node(clone, "05_preference_colour_scale").findall("s:rect", NS):
            cell.set("y", "5")
    reject("old_distant_preference_scale", old_scale_position)
    reject("fingerprint_boundary_misalignment", lambda clone: node(clone, "fingerprint_amine_record").findall("s:rect", NS)[47].set("x", "91"))
    reject("saved_step_misalignment", lambda clone: node(clone, "steps_halide_record").find("s:line", NS).set("x2", "145"))
    def absent_order(clone):
        item = node(clone, "retrospective_order")
        parent = next(parent for parent in clone.iter() if item in list(parent))
        parent.remove(item)
    reject("absent_retrospective_order_label", absent_order)
    reject("overlong_annotation", lambda clone: setattr(node(clone, "question"), "text", "Which molecular record carries over here?"))
    reject("old_layout_dimensions", lambda clone: clone.set("height", "120mm"))
    reject("previous_layout_dimensions", lambda clone: clone.set("height", "110mm"))
    def visible_pair_code(clone):
        extra = copy.deepcopy(node(clone, "measured_label"))
        extra.set("id", "internal_pair_code")
        extra.set("data-role", "axis")
        extra.text = "P1"
        clone.append(extra)
    reject("visible_internal_pair_code", visible_pair_code)
    reject("missing_cell_box_removed", lambda clone: next(cell for cell in node(clone, "fingerprint_reference").findall("s:rect", NS)
                                                           if cell.get("data-value") == "missing").set("stroke", "none"))
    def remove_missing_hatch(clone):
        hatch = clone.findall(".//*[@data-role='missing']")[0]
        identifier = hatch.get("id")
        next(parent for parent in clone.iter() if hatch in list(parent)).remove(hatch)
        metadata = clone.find("s:metadata", NS)
        meta = json.loads(metadata.text)
        meta["primitives"] = [item for item in meta["primitives"] if item["id"] != identifier]
        metadata.text = json.dumps(meta)
    reject("missing_cell_hatch_removed", remove_missing_hatch)
    reject("missing_hatch_assigned_to_finite_cell", lambda clone: clone.findall(".//*[@data-role='missing']")[0].set("data-cell-index", "0"))
    reject("wrong_preference_legend", lambda clone: setattr(node(clone, "preference_scale"), "text", "Yield (%)"))
    reject("misplaced_preference_tick", lambda clone: node(clone, "color_tick_0").set("x", "163"))
    def duplicate_structure(clone):
        extra = copy.deepcopy(node(clone, "target_halide"))
        for item in extra.iter():
            if item.get("id"):
                item.set("id", "duplicate_" + item.get("id"))
        extra.set("id", "known_halide")
        clone.append(extra)
    reject("duplicated_middle_structure", duplicate_structure)
    def swap_structure_roles(clone):
        halide, amine = node(clone, "target_halide"), node(clone, "target_amine")
        halide.set("id", "target_amine")
        amine.set("id", "target_halide")
    reject("swapped_native_structure_roles", swap_structure_roles, native_geometry)
    return results


def overlap_negative_control(svg, pdf, meta):
    clone = copy.deepcopy(meta)
    box = next(row["bbox_mm"] for row in clone["texts"] if row["id"] == "amine_class")
    yy = (box[1] + box[3]) / 2
    route = next(row for row in clone["primitives"] if row["id"] == "halide_route")
    route["points"] = [[box[0] - 1, yy], [box[2] + 1, yy]]
    try:
        layout_check(svg, pdf, clone)
    except AssertionError:
        return dict(test="annotation_route_collision", rejected=True)
    raise AssertionError("negative overlap control accepted")


def scale_overlap_negative_control(svg, pdf, meta):
    clone = copy.deepcopy(svg)
    box = next(row["bbox_mm"] for row in meta["texts"] if row["id"] == "color_tick_0")
    for cell in node(clone, "05_preference_colour_scale").findall("s:rect", NS):
        cell.set("y", str(box[3] - .1))
    preference_scale_geometry(clone)
    try:
        layout_check(clone, pdf, meta)
    except AssertionError:
        return dict(test="colourbar_callout_collision", rejected=True)
    raise AssertionError("negative colourbar overlap control accepted")


def artifact_paths(root, journal, artifact=None):
    assert journal in ("science", "natcatal")
    if artifact is not None:
        return artifact, artifact
    base = root / "02_FIGURES/Journal_Formats/natcatal" if journal == "natcatal" else root / "02_FIGURES"
    shared = root / "04_SOURCE_DATA/Figure1"
    variant = base / "Figure1"
    if journal == "natcatal" and variant.exists():
        assert variant.is_dir(), "Figure 1 Source Data variant must be a directory"
        for duplicate in variant.rglob("*"):
            if duplicate.is_file():
                canonical = shared / duplicate.relative_to(variant)
                assert canonical.is_file() and sha(duplicate) == sha(canonical), (
                    "Figure 1 Source Data must be shared between journals", duplicate, canonical)
    return base / "Main", shared


def main():
    global ROOT, INPUT
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--qc-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--journal", choices=("science", "natcatal"), default="science")
    parser.add_argument("--values-only", action="store_true", help="Compatibility flag; the focused audit always checks values and layout")
    args = parser.parse_args()
    ROOT = args.root.resolve()
    INPUT = ROOT / "05_DATA/Analysis/figure_display/v6_figure1"
    if args.out and args.artifact_dir and args.out.resolve() != args.artifact_dir.resolve():
        parser.error("--out and --artifact-dir must identify the same directory")
    artifact = (args.artifact_dir or args.out)
    artifact = artifact.resolve() if artifact else None
    art, data = artifact_paths(ROOT, args.journal, artifact)
    qc = (args.qc_dir or ROOT / "08_REVIEW_AND_QC/v6_formal_20261028").resolve()
    qc.mkdir(parents=True, exist_ok=True)
    public = os.environ.get("P2_RELEASE_MODE") == "public"
    assert sha(INPUT / "records.json") == (PORTABLE_RECORD_SHA if public else RECORD_SHA)
    assert sha(INPUT / "INPUTS.json") == (PORTABLE_INPUTS_SHA if public else INPUTS_SHA)
    assert sha(INPUT / "RULE.md") == RULE_SHA
    saved = read(INPUT / "records.json")
    if public:
        assert saved["source"] == "02_FIGURES/Main_v6_draft/aminations_payload.json"
    assert saved["freeze"] == read(INPUT / "FREEZE.json")
    svg = ET.parse(art / "Figure1.svg").getroot()
    report = verify_svg(svg, saved)
    meta = report.pop("metadata")
    assert meta == read(data / "Figure1_readback.json")
    report["native_D20_geometry"] = native_geometry(svg)
    workbook_check(data / "Figure1_SourceData.xlsx", saved)
    report["source_workbook"] = dict(display_cells=285, contributor_rows=190, saved_steps=20, member_pairs=15,
                                     exact_saved_value_strings=True)
    report["layout"] = layout_check(svg, art / "Figure1.pdf", meta)
    report["negative_controls"] = negative_controls(svg, saved)
    report["negative_controls"].append(overlap_negative_control(svg, art / "Figure1.pdf", meta))
    report["negative_controls"].append(scale_overlap_negative_control(svg, art / "Figure1.pdf", meta))
    payload = read(data / "figure1_payload.json")
    assert payload["schema"] == "v6_frozen_histories" and payload["source_records_sha256"] == RECORD_SHA
    assert payload["reference"]["pair"] == PAIR and payload["condition_permutation"] == saved["condition_permutation"]
    equal_saved(payload["reference"], saved["reference"])
    equal_saved(payload["records"], saved["records"])
    provenance = read(data / "Figure1_SourceData_provenance.json")
    assert provenance["figure"] == "Figure 1" and provenance["generated_by"] == "06_CODE/Figures/figure1.py"
    assert provenance["panels"] == {}
    for group in provenance["inputs"].values():
        for entry in group.values():
            assert sha(ROOT / entry["file"]) == entry["sha256"]
    for entry in provenance["outputs"].values():
        actual = (art if entry["file"].endswith((".svg", ".pdf", ".png")) else data) / Path(entry["file"]).name
        assert sha(actual) == entry["sha256"]
    checks = dict(retained_cells=dict(result="PASS", cells=285, steps=20),
                  independent_values=dict(result="PASS", records_sha256=RECORD_SHA, scientific_values_recalculated=False),
                  artists=dict(result="PASS", ordered_cells=285, saved_steps=20),
                  source_workbook=dict(result="PASS", **report["source_workbook"]),
                  native_geometry=dict(result="PASS", structures=report["native_D20_geometry"]),
                  journal=dict(result="PASS", unpaneled=True, journal_invariant=True),
                  export=dict(result="PASS", **report["layout"]),
                  provenance=dict(result="PASS", inputs=sum(len(group) for group in provenance["inputs"].values()),
                                  outputs=len(provenance["outputs"])))
    if not args.values_only:
        with tempfile.TemporaryDirectory(prefix="v6_figure1_", dir=qc) as folder:
            completed = subprocess.run([sys.executable, "-B", str(ROOT / "06_CODE/Figures/figure1.py"),
                                         "--out", folder, "--journal", args.journal], cwd=ROOT,
                                        capture_output=True, text=True, encoding="utf-8", errors="replace")
            assert completed.returncode == 0, completed.stdout + completed.stderr
            fresh = Path(folder)
            assert read(fresh / "Figure1_SourceData_provenance.json") == provenance
            for entry in provenance["outputs"].values():
                assert sha(fresh / Path(entry["file"]).name) == entry["sha256"]
            checks["fresh_rebuild"] = dict(result="PASS", outputs=len(provenance["outputs"]), deterministic=True)
    report.update(status="PASS", result="PASS", figure=1, journal=args.journal, checks=checks,
                  fault_injections=report["negative_controls"], records_sha256=RECORD_SHA,
                  records_input=dict(file=(INPUT / "records.json").relative_to(ROOT).as_posix(),
                                     sha256=sha(INPUT / "records.json")), scientific_values_recalculated=False)
    (qc / "Figure1_v6_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args.qc_dir:
        (qc / "Figure1_QA.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    # Preserve the complete v6 audit for archived artifacts; dispatch by actual artifact metadata.
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--root", type=Path, default=ROOT)
    probe.add_argument("--artifact-dir", type=Path)
    probe.add_argument("--out", type=Path)
    probe.add_argument("--journal", choices=("science", "natcatal"), default="science")
    parsed, _ = probe.parse_known_args()
    folder = parsed.artifact_dir or parsed.out or parsed.root / (
        "02_FIGURES/Main" if parsed.journal == "science" else "02_FIGURES/Journal_Formats/natcatal/Main")
    candidate = folder / "Figure1.svg"
    is_v7 = False
    if candidate.is_file():
        metadata = ET.parse(candidate).getroot().find("s:metadata", NS)
        is_v7 = metadata is not None and json.loads(metadata.text).get("version") == "v7_formal"
    if is_v7:
        from audit_figure1_v7 import main as v7_main
        v7_main()
    else:
        main()

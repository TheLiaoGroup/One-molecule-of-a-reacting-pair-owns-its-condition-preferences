"""Editable SVG and embedded-Arial PDF from exact frozen display records."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from openpyxl import Workbook
import pymupdf
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from structure_art import load as load_art
from figure_export import write_provenance

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "05_DATA/Analysis/figure_display/v6_figure1"
ART = ROOT / "02_FIGURES/Main"
DATA = ROOT / "04_SOURCE_DATA/Figure1"
RECORD_SHA = "1b97e9c409e1f6344dbe2be8c653664542a397b843c4d42bc8c6ee5f3f04eda6"
PORTABLE_RECORD_SHA = "9bf9a633f784eaba08e468b161951476d3152a28235f986b15ad4e8936e0bdfa"
INPUTS_SHA = "0838238d88bfd0e792b7fe0a462a35b499179c9cbf4778c399521afcb6848905"
PORTABLE_INPUTS_SHA = "1aa61e9d05d8b78107f56a03a9c02abb10e38359a7c2ff5f7f66ab1b3a72f0e6"
RULE_SHA = "50e5a7d1003d5d795cdf034f244d25a8324230a5ee00d3d6479763fce3c226c9"
FONT_SHA = {"Arial": "c9b76220a5be42ead4733611e417cd65c5fd8aeaa33eb56576ac378a37d130a1",
            "Arial-Bold": "766f06ac8761f82f25d032a220e89438f6064591af9915061f20b949efdedf69"}
PAIR = "Cc1cc(C)nc(N)c1 >> N#Cc1ccnc(Cl)c1"
NS = "http://www.w3.org/2000/svg"
INK = "http://www.inkscape.org/namespaces/inkscape"
PT = 72 / 25.4
WIDTH, HEIGHT = 180, 99
BAND_X, BAND_WIDTH, BAND_HEIGHT = 34, 112, 3.5
STEP_HEIGHT = 16
BLUE, ORANGE = "#0072B2", "#D55E00"
HEAT = LinearSegmentedColormap.from_list("v6_preference", ["#5A5A5A", "#FFFFFF", "#D9B748"])
ET.register_namespace("", NS)
ET.register_namespace("inkscape", INK)
MASTERS = {"halide": "02_FIGURES/ChemDraw/v5/f1/h2.svg",
           "amine": "02_FIGURES/ChemDraw/v5/f1/a2.svg"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def frozen():
    public = os.environ.get("P2_RELEASE_MODE") == "public"
    assert sha(INPUT / "records.json") == (PORTABLE_RECORD_SHA if public else RECORD_SHA)
    assert sha(INPUT / "INPUTS.json") == (PORTABLE_INPUTS_SHA if public else INPUTS_SHA)
    assert sha(INPUT / "RULE.md") == RULE_SHA
    records = json.loads((INPUT / "records.json").read_text(encoding="utf-8"))
    if public:
        assert records["source"] == "02_FIGURES/Main_v6_draft/aminations_payload.json"
    freeze = json.loads((INPUT / "FREEZE.json").read_text(encoding="utf-8"))
    assert records["freeze"] == freeze
    assert records["reference"]["pair"] == PAIR
    assert [row["n_pairs"] for row in records["records"]] == [6, 9]
    assert [row["r"] for row in records["records"]] == [0.3025779035389302, -0.10458708553363207]
    assert all(PAIR not in row["member_pairs"] for row in records["records"])
    return records


def fonts():
    for name, digest in FONT_SHA.items():
        if name not in pdfmetrics.getRegisteredFontNames():
            candidates = (Path(entry.fname) for entry in font_manager.fontManager.ttflist if entry.name == "Arial")
            path = next((item for item in candidates if item.is_file() and sha(item) == digest), None)
            if path is None:
                raise FileNotFoundError(f"Released Figure 1 font required: {name}, SHA-256 {digest}")
            pdfmetrics.registerFont(TTFont(name, str(path)))


class Scene:
    def __init__(self):
        self.svg = ET.Element(f"{{{NS}}}svg", width=f"{WIDTH}mm", height=f"{HEIGHT}mm",
                              viewBox=f"0 0 {WIDTH} {HEIGHT}")
        self.el(self.svg, "title").text = "The halide's molecular history follows the preference of a new pair"
        self.el(self.svg, "desc").text = (
            "The measured pair is treated retrospectively as a new pair. Both molecular histories exclude it. "
            "All three fingerprints use its saved descending measured condition order.")
        self.el(self.svg, "rect", x=0, y=0, width=WIDTH, height=HEIGHT, fill="white")
        self.texts, self.primitives, self.structures = [], [], []

    @staticmethod
    def el(parent, tag, **attributes):
        return ET.SubElement(parent, f"{{{NS}}}{tag}",
                             {key.replace("_", "-"): str(value) for key, value in attributes.items()})

    def layer(self, name):
        group = self.el(self.svg, "g", id=name)
        group.set(f"{{{INK}}}groupmode", "layer")
        group.set(f"{{{INK}}}label", name)
        return group

    def text(self, group, x, y, text, size=7, color="#222222", anchor="start", bold=False, id=None,
             role="annotation"):
        attrs = dict(x=x, y=y, font_family="Arial", font_size=size / PT, fill=color,
                     text_anchor=anchor, font_weight="bold" if bold else "normal", data_role=role)
        if id:
            attrs["id"] = id
        node = self.el(group, "text", **attrs)
        node.text = text
        font = "Arial-Bold" if bold else "Arial"
        width = pdfmetrics.stringWidth(text, font, size) / PT
        left = x - width * ({"start": 0, "middle": .5, "end": 1}[anchor])
        ascent, descent = pdfmetrics.getAscentDescent(font, size)
        self.texts.append(dict(id=id, text=text, fontsize=size, family=["Arial"], weight="bold" if bold else "normal",
                               color=color, role=role, bbox_mm=[left, y - ascent / PT, left + width, y - descent / PT]))
        return node

    def cited(self, group, x, y, text, citation, id):
        self.text(group, x, y, text, id=id)
        width = pdfmetrics.stringWidth(text, "Arial", 7) / PT
        self.text(group, x + width + .12, y - 1.0, citation, size=5, id=id + "_citation", role="citation")

    def line(self, group, points, color="#777777", width=.18, arrow=False, id=None, role="route"):
        attrs = dict(points=" ".join(f"{x},{y}" for x, y in points), fill="none", stroke=color,
                     stroke_width=width, data_role=role)
        if id:
            attrs["id"] = id
        node = self.el(group, "polyline", **attrs)
        self.primitives.append(dict(id=id, role=role, points=points, width_mm=width))
        if arrow:
            end, prior = np.asarray(points[-1]), np.asarray(points[-2])
            direction = (end - prior) / np.linalg.norm(end - prior)
            normal = np.asarray([-direction[1], direction[0]])
            corners = [end, end - 1.2 * direction + .45 * normal, end - 1.2 * direction - .45 * normal]
            self.el(group, "polygon", points=" ".join(f"{x},{y}" for x, y in corners), fill=color)
        return node

    def structure(self, group, kind, x, y, id):
        path = ROOT / MASTERS[kind]
        art = load_art(path)
        scale = 7 / art["label_px"] / PT
        node = self.el(group, "g", id=id, data_master=MASTERS[kind], data_master_sha256=sha(path),
                       data_atom_pt=7, data_native_uniform_scale=scale)
        for polygon in art["polygons"]:
            coords = polygon * scale + (x, y)
            self.el(node, "polygon", points=" ".join(f"{a:.9f},{b:.9f}" for a, b in coords),
                    fill="black", stroke="black", stroke_width=.08 / PT)
        for i, atom in enumerate(art["labels"]):
            self.text(node, x + atom["x"] * scale, y + atom["y"] * scale, atom["text"],
                      size=7 * atom["scale"], color="black", id=f"{id}_atom_{i}", role="atom")
        self.structures.append(dict(id=id, master=MASTERS[kind], sha256=sha(path), atom_pt=7,
                                    scale=scale, bbox_mm=[x, y, x + art["width"] * scale, y + art["height"] * scale]))

    def band(self, group, key, values, saved, x, y, width=BAND_WIDTH, height=BAND_HEIGHT):
        band = self.el(group, "g", id="fingerprint_" + key, data_width_mm=width, data_height_mm=height,
                       data_pair=PAIR if key == "reference" else "mean of other productive partners")
        for i, original in enumerate(saved["condition_permutation"]):
            value = values[original]
            attrs = dict(x=x + i * width / 95, y=y, width=width / 95 + .002, height=height,
                         fill=color(value), data_condition=saved["sorted_condition_order"][i],
                         data_original_index=original, data_value="missing" if not math.isfinite(value) else repr(value))
            cell = self.el(band, "rect", **attrs)
            if not math.isfinite(value):
                cell.set("stroke", "#B8B8B8")
                cell.set("stroke-width", ".10")
                self.line(band, [(float(cell.get("x")), y + height),
                                 (float(cell.get("x")) + width / 95, y)], color="#999999", width=.10,
                          id=f"missing_{key}_{i}", role="missing").set("data-cell-index", str(i))
        return band

    def steps(self, group, record, x, y, color):
        node = self.el(group, "g", id="steps_" + record["key"], data_ymin=-13, data_ymax=13,
                       data_axis_height_mm=STEP_HEIGHT, data_width_mm=BAND_WIDTH)
        self.line(node, [(x, y), (x, y + STEP_HEIGHT)], width=.14, color="#999999", role="axis")
        self.line(node, [(x, y + STEP_HEIGHT / 2), (x + BAND_WIDTH, y + STEP_HEIGHT / 2)],
                  width=.10, color="#AAAAAA", role="zero")
        for value in [-13, 0, 13]:
            self.text(node, x - 1.5, y + (13 - value) / 26 * 16 + .60,
                      str(value).replace("-", "\u2212") if value <= 0 else f"+{value}",
                      size=5, anchor="end", id=f"{record['key']}_tick_{value}", role="axis")
        previous = None
        for i, segment in enumerate(record["segments"]):
            x1 = x + segment["positions"][0] / 95 * BAND_WIDTH
            x2 = x + (segment["positions"][-1] + 1) / 95 * BAND_WIDTH
            yy = y + (13 - segment["mean"]) / 26 * 16
            self.el(node, "line", x1=x1, x2=x2, y1=yy, y2=yy, stroke=color, stroke_width=.23,
                    data_segment=i, data_mean=repr(segment["mean"]))
            if previous is not None:
                self.el(node, "line", x1=x1, x2=x1, y1=previous, y2=yy, stroke=color, stroke_width=.23)
            previous = yy


def color(value):
    if not math.isfinite(value):
        return "#FFFFFF"
    rgba = HEAT((min(max(value, -30), 30) + 30) / 60)
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in rgba[:3])


def make_scene(saved):
    fonts()
    s = Scene()
    header = s.layer("01_p1_question")
    s.text(header, 5, 5.2, "Ar\u2013X + H\u2013N \u2192 Ar\u2013N", size=5, id="reaction", role="scheme")
    s.text(header, 90, 5.2, "New pair: which record carries over?", anchor="middle", bold=True, id="question")
    s.structure(header, "halide", 54, 10, "target_halide")
    s.text(header, 91, 17.5, "+", anchor="middle", id="plus", role="scheme")
    s.structure(header, "amine", 106, 9, "target_amine")

    history = s.layer("02_symmetric_molecular_histories")
    halide, amine = saved["records"]
    for record, title, partners, xx, hue in [(halide, "halide's record", "amines", 63, BLUE),
                                            (amine, "amine's record", "halides", 114, ORANGE)]:
        s.text(history, xx, 25.3, title, color=hue, anchor="middle", id=record["key"] + "_title")
        s.text(history, xx, 29.5, f"{record['n_pairs']} other {partners}",
               color=hue, anchor="middle", id=record["key"] + "_partners")
    s.line(history, [(29, 44), (29, 49.75)], color=BLUE, arrow=True, id="halide_route")
    s.line(history, [(148, 70), (148, 75.75)], color=ORANGE, arrow=True, id="amine_route")
    guide = "ligand guides: amine class"
    guide_x = 114 - pdfmetrics.stringWidth(guide, "Arial", 7) / PT / 2
    s.cited(history, guide_x, 33.8, guide, "1", "amine_class")

    comparison = s.layer("03_aligned_measured_and_history_fingerprints")
    s.text(comparison, BAND_X, 35.3, "Ordered by measured preference", size=5, id="retrospective_order")
    s.text(comparison, 3, 40.6, "new pair, measured", id="measured_label")
    s.band(comparison, "reference", saved["reference"]["pattern_points"], saved, BAND_X, 38)
    for record, title, band_y, step_y, hue in [(halide, "halide", 48, 54, BLUE),
                                              (amine, "amine", 74, 80, ORANGE)]:
        s.text(comparison, 3, band_y + 2.6, title, color=hue, id=record["key"] + "_row")
        s.band(comparison, record["key"], record["pattern_points"], saved, BAND_X, band_y)
        s.steps(comparison, record, BAND_X, step_y, hue)
        s.text(comparison, 151, step_y + 9, f"r = {record['r']:.2f}".replace("-", "\u2212"),
               color=hue, id=record["key"] + "_correlation")

    scale = s.layer("05_preference_colour_scale")
    for i in range(120):
        s.el(scale, "rect", x=150 + i * 28 / 120, y=33, width=28 / 120 + .002, height=1.8,
             fill=color(-30 + i / 119 * 60))
    for x, value, anchor in [(150, "\u221230", "start"), (164, "0", "middle"), (178, "+30", "end")]:
        s.text(scale, x, 31.3, value, size=5, anchor=anchor, id="color_tick_" + value, role="axis")
    s.text(scale, 164, 27.5, "Preference (pp)", size=5, anchor="middle", id="preference_scale", role="axis")
    return s


def render_pdf(scene, path):
    canvas = Canvas(str(path), pagesize=(WIDTH * PT, HEIGHT * PT), invariant=1, pageCompression=1,
                    initialFontName="Arial", initialFontSize=7)
    canvas.setTitle("Figure 1")
    canvas.setAuthor("")
    canvas.setCreator("06_CODE/Figures/figure1.py")
    for node in scene.svg.iter():
        tag = node.tag.split("}")[-1]
        attrs = node.attrib
        fill = attrs.get("fill", "none")
        stroke = attrs.get("stroke", "none")
        if tag not in ("rect", "line", "polyline", "polygon", "text"):
            continue
        if fill != "none":
            canvas.setFillColor(fill)
        if stroke != "none":
            canvas.setStrokeColor(stroke)
            canvas.setLineWidth(float(attrs.get("stroke-width", .18)) * PT)
        if tag == "text":
            size = float(attrs["font-size"]) * PT
            canvas.setFont("Arial-Bold" if attrs.get("font-weight") == "bold" else "Arial", size)
            x, y = float(attrs["x"]) * PT, (HEIGHT - float(attrs["y"])) * PT
            {"start": canvas.drawString, "middle": canvas.drawCentredString, "end": canvas.drawRightString}[
                attrs.get("text-anchor", "start")](x, y, node.text or "")
        elif tag == "rect":
            x, y, w, h = (float(attrs.get(key, 0)) for key in ("x", "y", "width", "height"))
            canvas.rect(x * PT, (HEIGHT - y - h) * PT, w * PT, h * PT,
                        fill=int(fill != "none"), stroke=int(stroke != "none"))
        elif tag == "line":
            canvas.line(float(attrs["x1"]) * PT, (HEIGHT - float(attrs["y1"])) * PT,
                        float(attrs["x2"]) * PT, (HEIGHT - float(attrs["y2"])) * PT)
        else:
            points = [[float(value) for value in pair.split(",")] for pair in attrs["points"].split()]
            line = canvas.beginPath()
            line.moveTo(points[0][0] * PT, (HEIGHT - points[0][1]) * PT)
            for x, y in points[1:]:
                line.lineTo(x * PT, (HEIGHT - y) * PT)
            if tag == "polygon":
                line.close()
            canvas.drawPath(line, fill=int(fill != "none"), stroke=int(stroke != "none"))
    canvas.showPage()
    canvas.save()


def write_workbook(path, saved):
    workbook = Workbook()
    workbook.remove(workbook.active)
    tables = {}
    tables["Display_cells"] = [["fingerprint", "display_position", "condition", "original_index", "preference_pp", "fill", "exact_saved_value"]]
    for key, values in [("reference", saved["reference"]["pattern_points"])] + [
            (row["key"], row["pattern_points"]) for row in saved["records"]]:
        for i, original in enumerate(saved["condition_permutation"]):
            value = values[original]
            tables["Display_cells"].append([key, i, saved["sorted_condition_order"][i], original,
                                            value if math.isfinite(value) else None, color(value), repr(value)])
    tables["Record_contributors"] = [["record", "display_position", "condition", "contributors"]]
    tables["Saved_steps"] = [["record", "group", "positions_zero_based", "saved_mean_pp", "finite_conditions", "exact_saved_mean"]]
    tables["Member_pairs"] = [["record", "member_id", "pair", "excludes_P1"]]
    for row in saved["records"]:
        for i, original in enumerate(saved["condition_permutation"]):
            tables["Record_contributors"].append([row["key"], i, saved["sorted_condition_order"][i], row["contributor_counts"][original]])
        for i, segment in enumerate(row["segments"]):
            tables["Saved_steps"].append([row["key"], i, ",".join(map(str, segment["positions"])), segment["mean"], segment["finite_conditions"], repr(segment["mean"])])
        for member, pair in zip(row["member_ids"], row["member_pairs"]):
            tables["Member_pairs"].append([row["key"], member, pair, pair != PAIR])
    tables["Printed_numbers"] = [["quantity", "value", "meaning"],
        ["halide record r", saved["records"][0]["r"], "unbinned Pearson r against P1"],
        ["amine record r", saved["records"][1]["r"], "unbinned Pearson r against P1"],
        ["shared conditions per record", 93, "saved n_common"], ["conditions", 95, "shared display columns"],
        ["halide partners", 6, "other productive amines; P1 excluded"],
        ["amine partners", 9, "other productive halides; P1 excluded"],
        ["step lower pp", -13, "shared step axis"], ["step upper pp", 13, "shared step axis"],
        ["colour lower pp", -30, "clipped cell colour scale"], ["colour upper pp", 30, "clipped cell colour scale"],
        ["measurements", 177039, "full primary dataset"], ["screens", 15, "full primary dataset"],
        ["halide full experiment mean pairwise r", .27, "separate seven-amine experiment"],
        ["amine full experiment mean pairwise r", .07, "separate ten-halide experiment"],
        ["halide owns", "10/10", "halide-containing screens"], ["explained preference fold", 3, "outside-screen transfer"]]
    tables["Provenance"] = [["field", "value"], ["records_sha256", RECORD_SHA], ["rule_sha256", RULE_SHA],
                             ["P1", PAIR], ["scientific_values_recalculated", False],
                             ["order", "single saved descending P1 order; missing last"],
                             ["input", "05_DATA/Analysis/figure_display/v6_figure1/records.json"]]
    for name, rows in tables.items():
        sheet = workbook.create_sheet(name)
        for row in rows:
            sheet.append(row)
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = cell.font.copy(bold=True)
    workbook.properties.created = workbook.properties.modified = __import__("datetime").datetime(2000, 1, 1)
    workbook.save(path)
    with ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    properties = ET.fromstring(entries["docProps/core.xml"])
    for name in ("created", "modified"):
        properties.find("{http://purl.org/dc/terms/}" + name).text = "2000-01-01T00:00:00Z"
    entries["docProps/core.xml"] = ET.tostring(properties, encoding="utf-8")
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, value in sorted(entries.items()):
            info = ZipInfo(name, date_time=(2000, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, value)


def generate(out=None, journal="science"):
    saved = frozen()
    scene = make_scene(saved)
    artdir, datadir = (Path(out), Path(out)) if out else (ART, DATA)
    artdir.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)
    metadata = dict(figure=1, version="v6_formal", dimensions_mm=[WIDTH, HEIGHT], width_mm=WIDTH, height_mm=HEIGHT,
                    panels=[], panel_titles={}, journal_invariant=True, target_pair=PAIR, reference_pair=PAIR,
                    source_records_sha256=RECORD_SHA, rule_sha256=RULE_SHA, scientific_values_recalculated=False,
                    condition_permutation=saved["condition_permutation"], sorted_condition_order=saved["sorted_condition_order"],
                    record_summaries=[{key: row[key] for key in ("key", "n_pairs", "r", "n_common")} for row in saved["records"]],
                    step_axis_limit_pp=13, colour_range_pp=[-30, 30], global_atom_pt=7, master_bond_pt=14.4,
                    texts=scene.texts, text_styles=scene.texts, structures=scene.structures, primitives=scene.primitives)
    scene.el(scene.svg, "metadata").text = json.dumps(clean(metadata), ensure_ascii=True, sort_keys=True)
    ET.ElementTree(scene.svg).write(artdir / "Figure1.svg", encoding="utf-8", xml_declaration=True)
    render_pdf(scene, artdir / "Figure1.pdf")
    with pymupdf.open(artdir / "Figure1.pdf") as document:
        document[0].get_pixmap(dpi=600, alpha=False).save(artdir / "Figure1.png")
    write_workbook(datadir / "Figure1_SourceData.xlsx", saved)
    (datadir / "Figure1_readback.json").write_text(json.dumps(clean(metadata), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    payload = dict(schema="v6_frozen_histories", records_input="05_DATA/Analysis/figure_display/v6_figure1/records.json",
                   source_records_sha256=RECORD_SHA, scientific_values_recalculated=False,
                   reference=saved["reference"], records=saved["records"],
                   condition_permutation=saved["condition_permutation"], sorted_condition_order=saved["sorted_condition_order"])
    (datadir / "figure1_payload.json").write_text(json.dumps(clean(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    inputs = list(INPUT.iterdir()) + [ROOT / value for value in MASTERS.values()]
    inputs += [(ROOT / value).with_suffix(".cdxml") for value in MASTERS.values()]
    write_provenance(1, inputs, actual_dir=out, panels={}, outputs={"payload": DATA / "figure1_payload.json"})
    print(json.dumps(dict(figure=1, dimensions_mm=[WIDTH, HEIGHT], source_sha256=RECORD_SHA,
                          records_input=dict(file=(INPUT / "records.json").relative_to(ROOT).as_posix(),
                                             sha256=sha(INPUT / "records.json")),
                          scientific_values_recalculated=False, journal_invariant=True, output=str(artdir)), indent=2))


def cli():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--journal", choices=("science", "natcatal"), default=os.environ.get("P2_FIGURE_JOURNAL", "science"))
    args = parser.parse_args()
    generate(args.out, args.journal)

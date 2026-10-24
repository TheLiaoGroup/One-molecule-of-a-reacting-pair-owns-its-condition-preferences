"""Independent four-figure checks, isolated regeneration and targeted controls."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageChops
import pymupdf

import audit_figure_v3_common as previous
import audit_main_v5_values as values

PANELS = {1: list("ABCDE"), 2: list("ABC"), 3: list("AB"), 4: list("ABC")}
HEIGHTS = {1: 215., 2: 215., 3: 214., 4: 198.}


def read_source_tables(path):
    return values.read_stored_workbook(path)


def check_caption(number, manuscript, readback):
    match = re.search(rf"(?m)^\*\*Fig\. {number}\.\s+", manuscript)
    assert match, f"Figure {number}: missing Science caption"
    caption = manuscript[match.start():].split("\n\n", 1)[0]
    markers = list(re.finditer(r"(?:\*\*|[.!?])\s+\(([A-Z])\)\s*", caption))
    assert [m[1] for m in markers] == PANELS[number], f"Figure {number}: caption panel inventory/order"
    drawn = [item["text"] for item in readback["texts"] if (item.get("gid") or "").startswith("panel_")]
    assert drawn == PANELS[number], f"Figure {number}: actual panel inventory/order"
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(caption)
        assert caption[marker.end():end].strip(), f"Figure {number}: empty caption panel {marker[1]}"
    if number == 1:
        assert "triangle" in caption.lower() and "thin vertical line" not in caption.lower(), \
            "Figure 1: caption must identify actual best-condition triangles"
    return dict(panels=PANELS[number], caption_role="conclusions and reading instructions",
                exact_short_title_first_clause_required=False)


def check_text(number, readback):
    corrected = copy.deepcopy(readback)
    for correction in readback.get("d21", {}).get("annotation_text_bbox_corrections", []):
        matches = [item for item in corrected["texts"] if item.get("gid") == correction["gid"]
                   and item["text"] == correction["text"] and item["bbox"] == correction["artist_bbox"]]
        assert len(matches) == 1, "Annotation text-only bbox does not identify saved artist"
        matches[0]["bbox"] = correction["text_bbox"]
    return previous.check_text(number, corrected)


def check_d21(number, readback):
    evidence = readback.get("d21")
    assert isinstance(evidence, dict), f"Figure {number}: missing live D21 evidence"
    assert evidence.get("measurement_version") == 3 and evidence.get("criterion_pt") == 2., \
        f"Figure {number}: D21 rule changed"
    assert evidence.get("figure") == number, f"Figure {number}: D21 wrong figure"
    rows = evidence.get("rows", [])
    assert rows, f"Figure {number}: D21 has no measured annotations"
    expected = Counter((item.get("gid"), item["text"]) for item in readback["texts"]
                       if item.get("is_d21_annotation"))
    if expected:
        assert expected == Counter((row.get("gid"), row["text"]) for row in rows), \
            f"Figure {number}: D21 annotation inventory changed"
    failures = [row for row in rows if any(row.get(field) is not None and row[field] < 2. - 1e-6
                                         for field in ("graphic_gap_pt", "text_gap_pt"))]
    assert not failures and not evidence.get("failures") and evidence.get("status") == "PASS", \
        f"Figure {number}: D21 minimum clearance failed: {failures}"
    for field, aggregate in (("graphic_gap_pt", "minimum_graphic_gap_pt"),
                             ("text_gap_pt", "minimum_text_gap_pt")):
        actual = min((row[field] for row in rows if row.get(field) is not None), default=None)
        assert actual == evidence[aggregate], f"Figure {number}: D21 aggregate differs from rows"
    return dict(criterion_pt=2., annotations=len(rows), status="PASS",
                minimum_graphic_gap_pt=evidence["minimum_graphic_gap_pt"],
                minimum_text_gap_pt=evidence["minimum_text_gap_pt"])


def check_dimensions(number, root, readback):
    with pymupdf.open(root / f"02_FIGURES/Main/Figure{number}.pdf") as doc:
        assert len(doc) == 1, "Main figure PDF is not one page"
        page = doc[0]
        assert abs(page.rect.width / 72 * 25.4 - 180) < .01
        assert abs(page.rect.height / 72 * 25.4 - HEIGHTS[number]) < .01
        assert abs(readback["width_mm"] - 180) < 1e-6 and abs(readback["height_mm"] - HEIGHTS[number]) < 1e-6
        fonts = page.get_fonts()
        assert fonts and all(f[1] == "ttf" and "Arial" in f[3] for f in fonts), "Unembedded/non-Arial PDF fonts"
        assert page.get_drawings() and not page.get_images(), "Main PDF is not native vector"
    with Image.open(root / f"02_FIGURES/Main/Figure{number}.png") as image:
        assert abs(image.width - 180 / 25.4 * 600) < 1.1
        assert abs(image.height - HEIGHTS[number] / 25.4 * 600) < 1.1
        assert all(round(value) == 600 for value in image.info.get("dpi", (0, 0))), "PNG DPI differs"
        bounds = ImageChops.difference(image.convert("RGB"), Image.new("RGB", image.size, "white")).getbbox()
        assert bounds and min(bounds[0], bounds[1], image.width - bounds[2], image.height - bounds[3]) > 1, "Blank or edge-clipped PNG"
        pixels = image.size
    svg = ET.parse(root / f"02_FIGURES/Main/Figure{number}.svg")
    assert any(el.tag.endswith("}text") for el in svg.iter()), "SVG text converted to paths"
    return dict(width_mm=180., height_mm=HEIGHTS[number], png_pixels=pixels,
                dpi=600, native_vector_pdf=True, embedded_Arial_fonts=sorted({f[3].split("+")[-1] for f in fonts}))


def check_provenance(number, root, fresh, trace):
    result = previous.check_provenance(number, root, fresh, trace)
    if number == 4:
        path = root / "04_SOURCE_DATA/Figure4/Figure4_SourceData_provenance.json"
        prov = json.loads(path.read_text(encoding="utf-8"))
        outputs = {entry["file"] for entry in prov["outputs"].values()}
        names = ("SetA_owner_block_bootstrap_draws.csv", "SetA_whole_array_bootstrap_draws.csv")
        assert all(f"04_SOURCE_DATA/Figure4/{name}" in outputs for name in names), "Bootstrap owner is not Figure4"
    return result


def self_tests(number, root, tables, readback, manuscript=None):
    tests = []
    if manuscript is not None:
        changed = manuscript.replace(f"**Fig. {number}.", f"**Fig. {number + 10}.", 1)
        tests.append(previous.must_fail("missing caption", f"Figure {number}: missing Science caption",
                                       lambda: check_caption(number, changed, readback)))
    corrupted = copy.deepcopy(readback)
    ordinary = [item for item in corrupted["texts"] if previous.structure_group(item) is None]
    assert len(ordinary) >= 2, "Overlap control needs two ordinary text records"
    ordinary[1]["bbox"] = list(ordinary[0]["bbox"])
    tests.append(previous.must_fail("ordinary text overlap", f"Figure {number}: text bboxes overlap",
                                   lambda: check_text(number, corrupted)))
    corrupted_font = copy.deepcopy(readback)
    corrupted_font["texts"][0]["family"] = ["Times New Roman"]
    tests.append(previous.must_fail("non-Arial text", f"Figure {number}: non-Arial text",
                                   lambda: check_text(number, corrupted_font)))
    corrupted_d21 = copy.deepcopy(readback)
    corrupted_d21["d21"]["rows"][0]["graphic_gap_pt"] = 0.
    tests.append(previous.must_fail("annotation too near data", f"Figure {number}: D21 minimum clearance failed",
                                   lambda: check_d21(number, corrupted_d21)))
    tests.extend(values.fault_tests(number, root, tables, readback))
    assert len(tests) >= 4 and all(test["rejected"] for test in tests), "A targeted control lacks rejection evidence"
    return tests


def run(number):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--qc-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--values-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    qc = (args.qc_dir or root / "08_REVIEW_AND_QC/codex_formal_v5_20261022").resolve()
    qc.mkdir(parents=True, exist_ok=True)
    source = (args.artifact_dir or root / f"04_SOURCE_DATA/Figure{number}").resolve()
    report = dict(figure=number, result="FAIL", status="FAIL", phase="baseline", checks={})
    fresh = None
    try:
        if not args.values_only:
            fresh = Path(tempfile.mkdtemp(prefix=f"figure{number}_fresh_", dir=qc))
            trace = fresh / "audit_open_trace.json"
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
            process = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", previous.TRACE_CHILD,
                                      str(root / f"06_CODE/Figures/figure{number}.py"), str(fresh), str(trace)],
                                     cwd=root, env=env, capture_output=True, text=True, encoding="utf-8")
            (fresh / "generator.stdout.log").write_text(process.stdout, encoding="utf-8")
            (fresh / "generator.stderr.log").write_text(process.stderr, encoding="utf-8")
            assert process.returncode == 0, f"Fresh generator exit {process.returncode}: {process.stderr[-4000:]}"
            report["checks"]["provenance"] = check_provenance(number, root, fresh, trace)
        rb = json.loads((source / f"Figure{number}_readback.json").read_text(encoding="utf-8"))
        tables = read_source_tables(source / f"Figure{number}_SourceData.xlsx")
        report["checks"]["excel_reading"] = previous.check_excel_reading()
        report["checks"]["independent_values"] = values.audit_tables(number, root, tables, source)
        report["checks"]["independent_artists"] = values.audit_artists(number, root, tables, rb)
        report["checks"]["canonical_values"] = values.audit_current_tables(number, root, tables, rb)
        report["checks"]["text"] = check_text(number, rb)
        report["checks"]["d21"] = check_d21(number, rb)
        manuscript = None
        if not args.values_only:
            manuscript = (root / "01_MANUSCRIPT/Manuscript_Nature.md").read_text(encoding="utf-8")
            report["checks"]["caption"] = check_caption(number, manuscript, rb)
            report["checks"]["dimensions"] = check_dimensions(number, root, rb)
            report["checks"]["labels"] = previous.check_labels(root, tables)
        report["baseline"] = "PASS"
        if args.self_test or args.values_only:
            report["phase"] = "fault controls after passing baseline"
            report["fault_injections"] = self_tests(number, root, tables, rb, manuscript)
        report.update(status="PASS", result="PASS", phase="complete")
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    destination = qc / f"Figure{number}_v5_audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("figure", "result", "phase")}, ensure_ascii=False))
    if report["status"] != "PASS":
        print(report["error"])
        raise SystemExit(1)
    if fresh is not None:
        assert previous.inside(fresh.resolve(), qc.resolve())
        shutil.rmtree(fresh)
    return report

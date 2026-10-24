"""Independent file, provenance and layout checks for the five main figures."""
from __future__ import annotations

import argparse
import ast
import copy
import importlib
import itertools
import io
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import pandas as pd
from PIL import Image, ImageChops
import pymupdf


TRACE_CHILD = r'''
import json,os,runpy,sys
from pathlib import Path
script,out,trace=map(Path,sys.argv[1:])
events=set()
active=[True]
def hook(event,args):
    if active[0] and event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.fsdecode(args[0])).resolve()
        mode=args[1] if len(args)>1 and isinstance(args[1],str) else ''
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        write=any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        events.add((str(path),'w' if write else 'r'))
sys.addaudithook(hook)
sys.path.insert(0,str(script.parent))
sys.argv=[str(script),'--out',str(out)]
runpy.run_path(str(script),run_name='__main__')
active[0]=False
trace.write_text(json.dumps(sorted(events),indent=1),encoding='utf-8')
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source_tables(path):
    return pd.read_excel(path, sheet_name=None, keep_default_na=False, na_values=[""])


def check_excel_reading():
    sample = pd.DataFrame({"row_id": [1, 2], "ligand_family": ["None", None], "value": [1.0, None]})
    buffer = io.BytesIO()
    sample.to_excel(buffer, index=False, sheet_name="sample")
    buffer.seek(0)
    observed = read_source_tables(buffer)["sample"]
    assert observed.loc[0, "ligand_family"] == "None", "Literal None label was lost"
    assert pd.isna(observed.loc[1, "ligand_family"]), "A blank label was not missing"
    assert pd.isna(observed.loc[1, "value"]), "A blank numerical cell was not missing"
    return dict(literal_None_preserved=True, blank_label_missing=True, blank_number_missing=True)


def inside(path, directory):
    return path == directory or directory in path.parents


def plain(text):
    text = re.sub(r"S\$_\{?N\}?\$Ar|S<sub>N</sub>Ar", "SNAr", str(text))
    text = re.sub(r"\\mathit\{([A-Za-z])\}", r"\1", text)
    return " ".join(text.replace("$", "").split())


def check_caption(number, manuscript, readback):
    match = re.search(rf"(?m)^\*\*Fig\. {number}\.\s+", manuscript)
    assert match, f"Figure {number}: missing Science caption"
    caption = manuscript[match.start():].split("\n\n", 1)[0]
    markers = list(re.finditer(r"(?:\*\*|[.!?])\s+\(([A-Z])\)\s*", caption))
    titles = readback["panel_titles"]
    assert isinstance(titles, dict), f"Figure {number}: panel-title record must identify panels"
    assert [m[1] for m in markers] == list(titles), f"Figure {number}: caption panel inventory/order"
    visible = " ".join(plain(t["text"]) for t in readback["texts"])
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(caption)
        title = plain(caption[marker.end():end].split(".", 1)[0])
        assert title == plain(titles[marker[1]]), f"Figure {number}: caption panel {marker[1]} title differs"
        assert title in visible, f"Figure {number}: panel {marker[1]} title not drawn"
    return dict(panels=list(titles), titles={k: plain(v) for k, v in titles.items()})


def structure_group(text):
    gid = text.get("gid") or ""
    if text.get("chemical") and gid.startswith("focal-"):
        return gid
    if "|label" in gid and (gid.startswith("structure_") or "structure" in gid):
        return gid.split("|label", 1)[0]
    return None


def check_text(number, readback):
    texts = readback["texts"]
    assert texts, f"Figure {number}: no visible text"
    width, height = readback["width_px"], readback["height_px"]
    tolerance = 0.05 / readback["width_mm"] * width
    contacts = []
    for item in texts:
        x, y, w, h = item["bbox"]
        assert x >= -tolerance and y >= -tolerance and x + w <= width + tolerance and y + h <= height + tolerance, (
            f"Figure {number}: text outside canvas", item["text"], item["bbox"])
        assert item["family"] == ["Arial"], (f"Figure {number}: non-Arial text", item)
        size = item["fontsize"]
        chemical = structure_group(item)
        if abs(size - 6) < 1e-4:
            assert chemical and re.fullmatch(r"\d+|[+\-\u2212]", item["text"]), (
                f"Figure {number}: 6 pt is reserved for native chemical subscripts/charges", item)
        else:
            assert min(abs(size - 8), abs(size - 9)) < 1e-4, (f"Figure {number}: expected 8/9 pt", item)
        label = plain(item["text"])
        assert not re.search(r"\b(?:C|N|O)-(?:C|N|O)\b|Chan-Lam|->", label), f"Figure {number}: ASCII chemical connector: {label}"
        assert not re.search(r"(^|[-,(])n[-,]", label), f"Figure {number}: lowercase N locant: {label}"
        if "SNAr" in label:
            assert "S$_{N}$Ar" in item["text"], f"Figure {number}: SNAr N not subscripted"
    for left, right in itertools.combinations(texts, 2):
        ax, ay, aw, ah = left["bbox"]
        bx, by, bw, bh = right["bbox"]
        iw = min(ax + aw, bx + bw) - max(ax, bx)
        ih = min(ay + ah, by + bh) - max(ay, by)
        if iw <= 0 or ih <= 0:
            continue
        group = structure_group(left)
        if group and group == structure_group(right):
            contacts.append(dict(structure=group, text=[left["text"], right["text"]],
                                 intersection_mm=[iw / width * readback["width_mm"], ih / height * readback["height_mm"]]))
        else:
            raise AssertionError(f"Figure {number}: text bboxes overlap: {left['text']!r}, {right['text']!r}")
    return dict(texts=len(texts), ordinary_font_pt=[8, 9], native_subscript_or_charge_pt=6,
                ordinary_or_cross_structure_overlaps=0, outside_canvas=0,
                native_same_structure_bbox_contacts=contacts)


def check_dimensions(number, root, readback):
    approved = root / f"99_ARCHIVE/figures_v3_drafts/02_FIGURES/Main_v3_draft/Figure{number}.pdf"
    with pymupdf.open(approved) as reference, pymupdf.open(root / f"02_FIGURES/Main/Figure{number}.pdf") as document:
        assert len(reference) == len(document) == 1
        page, target = document[0], reference[0]
        assert max(abs(a - b) for a, b in zip(page.rect, target.rect)) < 0.01, "Approved PDF dimensions changed"
        assert abs(page.rect.width / 72 * 25.4 - readback["width_mm"]) < 0.01
        assert abs(page.rect.height / 72 * 25.4 - readback["height_mm"]) < 0.01
        fonts = page.get_fonts()
        assert fonts and all(f[1] == "ttf" and "Arial" in f[3] for f in fonts), f"Figure {number}: unembedded/non-Arial PDF fonts"
        assert page.get_drawings(), f"Figure {number}: no PDF vector drawing"
        assert not page.get_images(), f"Figure {number}: unexpected raster in vector PDF"
    with Image.open(root / f"02_FIGURES/Main/Figure{number}.png") as image:
        assert image.width == 4252 and abs(image.height - readback["height_mm"] / 25.4 * 600) < 1.1
        assert all(round(v) == 600 for v in image.info.get("dpi", (0, 0))), "PNG DPI differs"
        rgb = image.convert("RGB")
        bounds = ImageChops.difference(rgb, Image.new("RGB", image.size, "white")).getbbox()
        assert bounds and min(bounds[0], bounds[1], image.width - bounds[2], image.height - bounds[3]) > 1, "Blank or edge-clipped PNG"
        pixels = image.size
    svg = ET.parse(root / f"02_FIGURES/Main/Figure{number}.svg")
    assert any(el.tag.endswith("}text") for el in svg.iter()), "SVG text converted to paths"
    return dict(width_mm=readback["width_mm"], height_mm=readback["height_mm"], png_pixels=pixels,
                dpi=600, embedded_Arial_fonts=sorted({f[3].split("+")[-1] for f in fonts}), vector_pdf=True)


def check_labels(root, tables):
    source = ast.parse((root / "06_CODE/Figures/figure_common.py").read_text(encoding="utf-8"))
    labels = next(ast.literal_eval(node.value) for node in source.body if isinstance(node, ast.Assign)
                  and any(getattr(t, "id", None) == "ARRAY_LABELS" for t in node.targets))
    assert len(labels) == len(set(labels.values())) == 15
    checked = 0
    for frame in tables.values():
        if {"screen", "array"} <= set(frame):
            for screen, value in frame[["screen", "array"]].dropna().itertuples(index=False, name=None):
                if screen in labels:
                    assert plain(value) == labels[screen], f"Noncanonical array label {screen}: {value}"
                    checked += 1
        for column in frame:
            if column in {"name", "partner", "partner_i", "partner_j", "display_name"} or column.endswith("_name"):
                for value in frame[column].dropna():
                    assert not re.search(r"(^|[-,(])n[-,]", plain(value)), f"Lowercase N locant in {column}: {value}"
    return dict(canonical_arrays=15, checked_label_records=checked)


def flatten_inputs(provenance):
    entries = []
    for role in provenance["inputs"].values():
        assert isinstance(role, dict)
        for relative, entry in role.items():
            assert relative == entry["file"], "Input path key differs from file identity"
            entries.append(entry)
    assert len({entry["file"] for entry in entries}) == len(entries), "Duplicate provenance inputs"
    return entries


def check_provenance(number, root, fresh, trace):
    source = root / f"04_SOURCE_DATA/Figure{number}"
    prov_path = source / f"Figure{number}_SourceData_provenance.json"
    provenance = json.loads(prov_path.read_text(encoding="utf-8"))
    assert provenance["generated_by"] == f"06_CODE/Figures/figure{number}.py"
    inputs = flatten_inputs(provenance)
    outputs = list(provenance["outputs"].values())
    assert not ({e["file"] for e in inputs} & {e["file"] for e in outputs}), "An output is declared as its own scientific input"
    for entry in inputs + outputs:
        relative = Path(entry["file"])
        path = (root / relative).resolve()
        assert not relative.is_absolute() and inside(path, root), f"Nonrelative provenance path: {entry}"
        assert sha(path) == entry["sha256"], f"Current provenance hash differs: {entry['file']}"
    for entry in outputs:
        delivered = root / entry["file"]
        assert (fresh / delivered.name).read_bytes() == delivered.read_bytes(), f"Figure {number}: fresh bytes differ: {entry['file']}"
    assert (fresh / prov_path.name).read_bytes() == prov_path.read_bytes(), f"Figure {number}: fresh provenance differs"
    events = [(Path(path).resolve(), mode) for path, mode in json.loads(trace.read_text(encoding="utf-8"))]
    reads = {path for path, mode in events if mode == "r" and path.is_file() and not inside(path, fresh)}
    forbidden = [str(p) for p in reads if "sealed" in str(p).casefold() or (inside(p, root.parent) and not inside(p, root))]
    assert not forbidden, f"Other-project or sealed input opened: {forbidden}"
    project_tree = root.parent.parent
    writes = [str(path) for path, mode in events if mode == "w" and inside(path, project_tree) and not inside(path, fresh)]
    assert not writes, f"Fresh generation wrote in the project tree outside scratch: {writes}"
    external_writes = sorted({str(path) for path, mode in events if mode == "w" and not inside(path, project_tree)})
    science = {p.relative_to(root).as_posix() for p in reads if inside(p, root) and (
        p.relative_to(root).as_posix() == "06_CODE/Directional_Ownership/phase2_section7.py" or any(
            part in p.relative_to(root).parts for part in ("05_DATA", "04_SOURCE_DATA", "ChemDraw", "07_METHODS_AND_PROVENANCE", "10_PUBLIC_RELEASE_CANDIDATE")))}
    declared = {entry["file"] for entry in inputs}
    assert science == declared, f"Observed scientific inputs differ from declared: {sorted(science ^ declared)}"
    if number == 5:
        names = ("SetA_owner_block_bootstrap_draws.csv", "SetA_whole_array_bootstrap_draws.csv")
        assert all(f"04_SOURCE_DATA/Figure5/{name}" in {e['file'] for e in outputs} for name in names), "Bootstrap owner is not Figure5"
    return dict(exact_outputs=len(outputs) + 1, observed_scientific_inputs=sorted(science),
                no_project_tree_writes_outside_scratch=True, external_runtime_writes=external_writes,
                browser_native_file_reads_audited=False if number == 1 else None)


def must_fail(name, expected, action):
    try:
        action()
    except AssertionError as error:
        assert expected in str(error), f"Fault {name} rejected for an unrelated reason: {error}"
        return dict(fault=name, rejected=True, new_issue=str(error), target=expected)
    raise AssertionError("Planted fault accepted: " + name)


def self_tests(number, root, tables, readback, manuscript, module):
    title = next(iter(readback["panel_titles"].values()))
    title_plain = plain(title)
    changed = manuscript.replace(title_plain + ".", title_plain + " altered.", 1)
    assert changed != manuscript, "Caption fault was not inserted"
    tests = [must_fail("caption title word", f"Figure {number}: caption panel", lambda: check_caption(number, changed, readback))]
    drawing = copy.deepcopy(readback)
    item = next((item for item in drawing["texts"] if "C\u2013N" in item["text"]), drawing["texts"][0])
    item["text"] = item["text"].replace("C\u2013N", "C-N", 1) if "C\u2013N" in item["text"] else item["text"] + " C-N"
    tests.append(must_fail("ASCII C-N", f"Figure {number}: ASCII chemical connector", lambda: check_text(number, drawing)))
    drawing = copy.deepcopy(readback)
    ordinary = [item for item in drawing["texts"] if structure_group(item) is None]
    assert len(ordinary) >= 2, "Overlap fault requires two ordinary text records"
    ordinary[1]["bbox"] = list(ordinary[0]["bbox"])
    tests.append(must_fail("ordinary text overlap", f"Figure {number}: text bboxes overlap", lambda: check_text(number, drawing)))
    drawing = copy.deepcopy(readback)
    chemical = [item for item in drawing["texts"] if structure_group(item)]
    pair = next(((a, b) for a, b in itertools.combinations(chemical, 2)
                 if structure_group(a) != structure_group(b)), None)
    if pair is not None:
        pair[1]["bbox"] = list(pair[0]["bbox"])
        tests.append(must_fail("cross-structure text overlap", f"Figure {number}: text bboxes overlap", lambda: check_text(number, drawing)))
    drawing = copy.deepcopy(readback)
    drawing["texts"][0]["family"] = ["Times New Roman"]
    tests.append(must_fail("non-Arial text", f"Figure {number}: non-Arial text", lambda: check_text(number, drawing)))
    assert hasattr(module, "fault_tests"), "Independent data fault controls are required"
    tests.extend(module.fault_tests(number, root, tables, readback))
    assert tests and all(test.get("rejected") for test in tests), "A fault lacks rejection evidence"
    return tests


def run(number, module_name):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    qc = root / "08_REVIEW_AND_QC/codex_formal_v3_20261021"
    qc.mkdir(exist_ok=True)
    fresh = Path(tempfile.mkdtemp(prefix=f"figure{number}_fresh_", dir=qc))
    report = dict(figure=number, status="FAIL", phase="fresh generation", temporary=fresh.relative_to(root).as_posix())
    try:
        trace = fresh / "audit_open_trace.json"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
        process = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", TRACE_CHILD,
                                  str(root / f"06_CODE/Figures/figure{number}.py"), str(fresh), str(trace)],
                                 cwd=root, env=env, capture_output=True, text=True, encoding="utf-8")
        (fresh / "generator.stdout.log").write_text(process.stdout, encoding="utf-8")
        (fresh / "generator.stderr.log").write_text(process.stderr, encoding="utf-8")
        assert process.returncode == 0, f"Fresh generator exit {process.returncode}: {process.stderr[-4000:]}"
        report["phase"] = "baseline comparisons"
        report["provenance"] = check_provenance(number, root, fresh, trace)
        source = root / f"04_SOURCE_DATA/Figure{number}"
        readback = json.loads((source / f"Figure{number}_readback.json").read_text(encoding="utf-8"))
        manuscript = (root / "01_MANUSCRIPT/Manuscript_Nature.md").read_text(encoding="utf-8")
        tables = read_source_tables(source / f"Figure{number}_SourceData.xlsx")
        report["excel_reading_controls"] = check_excel_reading()
        report["caption"] = check_caption(number, manuscript, readback)
        report["text"] = check_text(number, readback)
        report["dimensions"] = check_dimensions(number, root, readback)
        report["labels"] = check_labels(root, tables)
        module = importlib.import_module(module_name)
        report["independent_values"] = module.audit_tables(number, root, tables)
        report["independent_artists"] = module.audit_artists(number, root, tables, readback)
        report["baseline"] = "PASS"
        if args.self_test:
            report["phase"] = "fault controls after passing baseline"
            report["fault_injections"] = self_tests(number, root, tables, readback, manuscript, module)
        report["status"] = "PASS"
        report["phase"] = "complete"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    destination = qc / f"figure{number}_v3_audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("figure", "status", "phase")}, ensure_ascii=False))
    if report["status"] != "PASS":
        print(report["error"])
        raise SystemExit(1)
    assert inside(fresh.resolve(), qc.resolve())
    shutil.rmtree(fresh)
    return report

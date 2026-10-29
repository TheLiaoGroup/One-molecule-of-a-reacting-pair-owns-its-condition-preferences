"""Rebuild the article's figures, supplementary figures and Source Data from the released P2 package and an explicitly verified external result, and compare them with the
released ones.

    python rebuild_all.py                      everything (about 15 minutes with 6 processes on the computer used for the study)
    python rebuild_all.py --jobs 4             fewer parallel processes
    python rebuild_all.py --only figure3 ed05  these generators only (no re-fit; only their outputs are compared)
    python rebuild_all.py --zhang FILE          also re-derive the provenance flags from the pinned Zhang reference file
                                               (the reference collection is not redistributed in this package)

Steps:
  1. every file of the package is checked against CHECKSUMS.sha256;
  2. task construction: the derived analysis tables are made again from the released records (data/) and checked against
     derived/ and derived/DERIVED_FILES.json;
  3. a work tree is built in _rebuild/tree: copies (never links) of the package files at the paths the analysis code reads;
  4. one font list is fixed for every process (environment/fonts.json);
  5. the five main and ten supplementary figure generators run, each in its own Python process, in public mode (P2_RELEASE_MODE=public): the hidden
     folds and draws are read from the released Source Data instead of the internal registry of the original analysis;
  6. Fig. S10's models are re-fitted from the measurements and the released folds and groups
     (code/Checks/verify_ed10_refit.py, fold registry blocked);
  7. the provenance flags are checked against the released records, and re-derived when --zhang is given;
  8. the statements of the text in results/TEXT_CLAIMS.csv and the publication use of every released measurement are
     recomputed;
  9. every rebuilt figure, Source Data file and provenance record is compared with the released one: figures and Source Data
     byte for byte; a provenance record field for field, its list of inputs naming the package files that were read;
 10. the owner labels of Fig. 4 are fixed again without the scored pairs, and the shares are re-estimated within each array's
     largest connected network of pairs (code/Checks/nested_owner.py, on the rebuilt Figure 4 Source Data); the result must
     equal results/directional/nested_owner_results.json;
 11. the per-array owner margins of table S1, the equal-role null, the SNAr class proxy and the SNAr nesting are
     recomputed from the measurements (code/Checks/role_null_checks.py, about 8 minutes); the result must equal
     results/directional/role_null_checks.json;
 12. the per-array tables and single values of the Supplementary Information that no figure prints (tables S5 and S7,
     Notes 7, 10 and 15) and the counts of Methods and Supplementary Note 3 are recomputed from the measurements and from the result
     files the earlier steps verify (code/Checks/si_tables_checks.py); the result must equal results/directional/si_tables_checks.json;
 13. the values that the figure generators and the Supplementary Information take from the full analysis run are recomputed from
     the measurements: Fig. 2A's interval of the median molecular share and the owner named by the likelihood estimator, and
     the frozen three-condition decision on Set A, planned from the training arrays and scored on the released Set A
     measurements (code/Checks/frozen_reads_recomputed.py, about 1 minute); the result must equal
     results/directional/frozen_reads_recomputed.json.
 14. five main-figure audits independently recompute the displayed values and check the actual series of this run's outputs
     (code/Checks/audit_figure1.py to audit_figure5.py, --values-only); retained computed input tables are checked against
     measurements rather than accepted solely from their hashes.
Every Python process of the rebuild runs under a file-access guard (code/release/guard/sitecustomize.py): it may read this
package, Python and system fonts, and write only under _rebuild/. Only this Python executable may start child processes;
Python network connections are refused and Python file openings are logged.
Writes _rebuild/REBUILD_REPORT.json and prints a summary; the exit code is 0 only if every step passed.
"""
import argparse
import importlib.util
import json
import os
import platform
import shutil
import site
import subprocess
import sys
import sysconfig
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(PKG / "code/release"))
import package_lib as L                                                      # noqa: E402
_spec = importlib.util.spec_from_file_location("p2_public_guard", PKG / "code/release/guard/sitecustomize.py")
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)                                              # installs nothing until install() is called

WORK, TREE, OUT = L.WORK, L.WORK / "tree", L.WORK / "out"
def allowed_read(extra=()):
    """This package, the Python installation and the system font directories."""
    import matplotlib
    from matplotlib import font_manager as fm
    roots = {str(PKG), sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, os.path.dirname(sys.executable),
             site.getusersitepackages(), matplotlib.get_data_path()}
    roots |= set(site.getsitepackages()) | {v for k, v in sysconfig.get_paths().items() if k in ("stdlib", "platstdlib", "purelib", "platlib")}
    fonts = set(getattr(fm, "MSUserFontDirectories", [])) | set(getattr(fm, "X11FontDirectories", [])) | set(getattr(fm, "OSXFontDirectories", []))
    if sys.platform == "win32":
        fonts.add(fm.win32FontDirectory())
    fonts |= {os.path.expanduser(x) for x in ("~/.fonts", "~/.local/share/fonts", "~/Library/Fonts")}
    return sorted(roots | fonts | {str(Path(x).resolve()) for x in extra})


def font_list():
    """One font list for every process of the rebuild: the Arial files the released figures were drawn with
    (environment/fonts.json) first, every other entry in path order. Without it, the choice between two installed files of the
    same Arial style would follow the order in which a fresh font list happened to read the font directories."""
    from matplotlib import font_manager as fm
    want = {f["sha256"] for f in json.loads((PKG / "environment/fonts.json").read_text(encoding="utf-8"))["used_by_the_released_figures"]}
    m = fm.FontManager()
    arial = {e.fname: L.sha256(e.fname) for e in m.ttflist if e.name == "Arial"}
    m.ttflist.sort(key=lambda e: (0 if arial.get(e.fname) in want else 1, os.path.normcase(e.fname)))
    m.afmlist.sort(key=lambda e: os.path.normcase(e.fname))
    fm.json_dump(m, WORK / "mplconfig" / f"fontlist-v{fm.FontManager.__version__}.json")
    missing = want - set(arial.values())
    return ([f"{len(missing)} Arial file(s) of the released figures not installed: the PDF files will embed other font programs"] if missing else []), dict(
        arial_files=len(arial), released_figure_fonts_installed=len(want - missing))


def step(name, report, fn):
    t0 = time.time()
    try:
        problems, info = fn()
    except Exception as exc:                                      # a step that cannot run has failed
        problems, info = [f"{type(exc).__name__}: {exc}"], {}
    report["steps"].append(dict(step=name, result="PASS" if not problems else "FAIL", problems=problems, seconds=round(time.time() - t0, 1), **info))
    print(f"{'PASS' if not problems else 'FAIL'}  {name}" + "".join(f"\n      {p}" for p in problems[:8]))
    return not problems


def checksums():
    listed = {}
    for line in (PKG / L.CHECKSUMS).read_text(encoding="utf-8").splitlines():
        h, f = line.split("  ", 1)
        listed[f] = h
    bad = [f"{f}: {'missing' if not (PKG / f).is_file() else 'changed'}" for f, h in listed.items() if not (PKG / f).is_file() or L.sha256(PKG / f) != h]
    present = {p.relative_to(PKG).as_posix() for p in PKG.rglob("*") if p.is_file() and "__pycache__" not in p.parts
               and not p.relative_to(PKG).as_posix().startswith("_rebuild/")
               and p.relative_to(PKG).parts[0] != ".git"}                       # a clone's repository metadata, not part of the release
    bad += [f"{f}: not listed in {L.CHECKSUMS}" for f in sorted(present - set(listed) - {L.CHECKSUMS})]
    return bad, dict(files=len(listed))


def task_construction():
    rec = json.loads((PKG / L.DERIVED_RECORD).read_text(encoding="utf-8"))
    mine = L.build_derived(WORK)
    bad = []
    for name, r in rec.items():
        m = mine.get(name)
        if m is None:
            bad.append(f"{name}: not made by this run")
            continue
        for k in ("inputs", "rows", "content_sha256", "transformation"):
            if m[k] != r[k]:
                bad.append(f"{name}: {k} differs from {L.DERIVED_RECORD}")
        if L.sha_bytes(L.payload(PKG / name)) != m["content_sha256"]:
            bad.append(f"{name}: the released file's content differs from this run's")
    bad += [f"{n}: made by this run but not released" for n in sorted(set(mine) - set(rec))]
    same_bytes = all(L.sha256(WORK / n) == L.sha256(PKG / n) for n in rec)
    bad += L.check_derived(rec, L.load_units(), where=WORK)
    return bad, dict(files=sorted(rec), byte_identical_to_released=same_bytes)


def run(cmd, log, env):
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=str(TREE), env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def generators(names, jobs, env):
    todo = [(n, s) for n, s in L.GENERATORS if n in names]
    journal = L.journal()

    def one(item):
        n, s = item
        out = OUT / n
        out.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-B", str(TREE / s), "--out", str(out)]
        if n.startswith("figure") or n == "figs29":
            command += ["--journal", journal]
        return n, run(command, WORK / "logs" / f"{n}.log", env)
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        codes = dict(ex.map(one, todo))
    bad = [f"{n}: exit code {c} (see _rebuild/logs/{n}.log)" for n, c in codes.items() if c != 0]
    return bad, dict(generators=sorted(codes))


def figure_audits(names, env):
    bad, records = [], {}
    qc = WORK / "figure_audits"
    qc.mkdir(parents=True, exist_ok=True)
    for name, _ in L.GENERATORS:
        if name not in names or not name.startswith("figure"):
            continue
        number = int(name.removeprefix("figure"))
        script = TREE / f"06_CODE/Checks/audit_figure{number}.py"
        code = run([sys.executable, "-B", str(script), "--values-only", "--root", str(TREE), "--qc-dir", str(qc),
                    "--artifact-dir", str(OUT / name), "--journal", L.journal()], WORK / "logs" / f"audit_{name}.log", env)
        path = qc / f"Figure{number}_v6_audit.json"
        if code != 0:
            bad.append(f"{name} independent audit: exit code {code} (see _rebuild/logs/audit_{name}.log)")
        if not path.is_file():
            bad.append(f"{name} independent audit: report missing")
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        records[name] = dict(result=record.get("result"), checks=len(record.get("checks", [])),
                             report=path.relative_to(WORK).as_posix())
        if record.get("result") != "PASS":
            bad.append(f"{name} independent audit: {record.get('result')}")
    return bad, dict(audits=records)


def refit(jobs, env):
    rec = OUT / "ED10_REFIT_VERIFICATION.json"
    c = run([sys.executable, "-B", str(TREE / L.REFIT), "--jobs", str(jobs), "--record", str(rec)], WORK / "logs" / "ed10_refit.log", env)
    r = json.loads(rec.read_text(encoding="utf-8")) if rec.is_file() else {}
    bad = [] if c == 0 and r.get("result") == "PASS" else [f"re-fit exit code {c}, result {r.get('result')} (see _rebuild/logs/ed10_refit.log)"]
    return bad, dict(cases=r.get("regimes", {}).get("cases"), associations=r.get("association", {}).get("arrays"))


def nested(env):
    rec = WORK / "nested_owner_results.json"
    c = run([sys.executable, "-B", str(TREE / L.NESTED), "--source-data", str(OUT / "figure5/Figure5_SourceData.xlsx"),
             "--folds", str(TREE / "05_DATA/Analysis/figure_display/Fig4a_folds.csv"), "--out", str(rec)], WORK / "logs" / "nested_owner.log", env)
    if c != 0 or not rec.is_file():
        return [f"exit code {c} (see _rebuild/logs/nested_owner.log)"], {}
    same = json.loads(rec.read_text(encoding="utf-8")) == json.loads((PKG / "results/directional/nested_owner_results.json").read_text(encoding="utf-8"))
    byte_identical = rec.read_bytes() == (PKG / "results/directional/nested_owner_results.json").read_bytes()
    r = json.loads(rec.read_text(encoding="utf-8"))
    return ([] if same and byte_identical else ["differs from results/directional/nested_owner_results.json"]), dict(
        byte_identical=byte_identical,
        records_moved=r["fig4a"]["records_moved"], targets_flipped=r["fig4b"]["flipped_targets"],
        largest_network_same_owner=r["components"]["largest_same_owner"])


def nullchk(env):
    rec = WORK / "role_null_checks.json"
    c = run([sys.executable, "-B", str(TREE / L.NULLCHK), "--out", str(rec)], WORK / "logs" / "role_null_checks.log", env)
    if c != 0 or not rec.is_file():
        return [f"exit code {c} (see _rebuild/logs/role_null_checks.log)"], {}
    r = json.loads(rec.read_text(encoding="utf-8"))
    same = r == json.loads((PKG / "results/directional/role_null_checks.json").read_text(encoding="utf-8"))
    return ([] if same else ["differs from results/directional/role_null_checks.json"]), dict(
        margins=len(r["margins"]), null_reaching_observed=sum(m["reaching_observed"] for m in r["equal_role_null"]["models"].values()))


def sitables(env):
    rec = WORK / "si_tables_checks.json"
    c = run([sys.executable, "-B", str(TREE / L.SITABLES), "--out", str(rec), "--mask", str(PKG / L.MASK),
             "--a1", str(PKG / "data/measurements_A1_analysis_arrays.csv.gz"), "--conditions", str(PKG / "data/conditions.csv")],
            WORK / "logs" / "si_tables_checks.log", env)
    if c != 0 or not rec.is_file():
        return [f"exit code {c} (see _rebuild/logs/si_tables_checks.log)"], {}
    r = json.loads(rec.read_text(encoding="utf-8"))
    same = r == json.loads((PKG / "results/directional/si_tables_checks.json").read_text(encoding="utf-8"))
    return ([] if same else ["differs from results/directional/si_tables_checks.json"]), dict(
        table_5_rows=len(r["supplementary_table_5"]), table_7_rows=len(r["supplementary_table_7"]))


def frozenreads(env):
    rec = WORK / "frozen_reads_recomputed.json"
    c = run([sys.executable, "-B", str(TREE / L.FROZENREADS), "--out", str(rec), "--set-a", str(PKG / "data/measurements_SA1_set_A.csv.gz")],
            WORK / "logs" / "frozen_reads_recomputed.log", env)
    if c != 0 or not rec.is_file():
        return [f"exit code {c} (see _rebuild/logs/frozen_reads_recomputed.log)"], {}
    r = json.loads(rec.read_text(encoding="utf-8"))
    same = r == json.loads((PKG / "results/directional/frozen_reads_recomputed.json").read_text(encoding="utf-8"))
    return ([] if same else ["differs from results/directional/frozen_reads_recomputed.json"]), dict(
        molecular_median_interval=r["figure1"]["molecular_median_interval"], decision_targets=r["e3"]["targets"])


def provenance(env, zhang):
    script = TREE / "06_CODE/Release/provenance_overlap_public.py"
    if zhang:
        out = WORK / "provenance"
        out.mkdir(exist_ok=True)
        c = run([sys.executable, "-B", str(script), "--zhang", str(zhang), "--out", str(out)],
                WORK / "logs" / "provenance.log", env)
        bad = [] if c == 0 else [f"exit code {c} (see _rebuild/logs/provenance.log)"]
        if not bad:
            if L.sha256(out / "P2_RECORD_PROVENANCE.csv") != L.sha256(PKG / L.PROVENANCE_FLAGS):
                bad.append("re-derived P2_RECORD_PROVENANCE.csv differs from the released one")
            a = json.loads((PKG / L.PROVENANCE_COUNTS).read_text(encoding="utf-8"))
            b = json.loads((out / "P2_ZHANG_OVERLAP_COUNTS_PUBLIC.json").read_text(encoding="utf-8"))
            bad += [f"counts: {k} differs" for k in ("measurements", "zhang_overlap", "cross_check") if a[k] != b[k]]
        return bad, dict(mode="re-derived from the released measurements and the Zhang file")
    c = run([sys.executable, "-B", str(script)], WORK / "logs" / "provenance.log", env)
    return ([] if c == 0 else [f"exit code {c} (see _rebuild/logs/provenance.log)"]), dict(
        mode="released records and Zhang flags checked against the released counts")


def claims_and_use(rebuilt_fig3):
    U = L.load_units()
    prov = rebuilt_fig3 if rebuilt_fig3.is_file() else PKG / "source_data/Figure3/Figure3_SourceData_provenance.json"
    values = L.text_claim_values(U, L.read_text_table(L.MASK), L.read_text_table(L.CONDITIONS), L.read_text_table(L.REAGENTS),
                                 L.read_text_table(L.PROVENANCE_FLAGS), json.loads(prov.read_text(encoding="utf-8")))
    bad = L.check_text_claims(values) + L.check_publication_use(U)
    return bad, dict(claims=sorted(values), figure3_provenance="rebuilt" if prov == rebuilt_fig3 else "released")


def compare(names):
    rows, bad, notes = [], [], []
    covered = set()
    for n, _ in L.GENERATORS:
        if n not in names:
            continue
        provs = list((OUT / n).glob("*_provenance.json"))
        if len(provs) != 1:
            bad.append(f"{n}: no provenance record written")
            continue
        new = json.loads(provs[0].read_text(encoding="utf-8"))
        owned = {L.delivered(value["file"]) for value in new["outputs"].values()}
        bad += [f"{name}: not regenerated by its owner {n}" for name in L.SUPPORT_OUTPUTS.get(n, []) if name not in owned]
        for key, v in new["outputs"].items():
            d = L.delivered(v["file"])
            if d is None:
                continue
            covered.add(d)
            status, note = L.same_payload(OUT / n / Path(v["file"]).name, PKG / d) if (PKG / d).is_file() else ("missing", "")
            rows.append(dict(generator=n, file=d, status=status, note=note))
            if status not in ("identical", "payload identical"):
                bad.append(f"{d}: {status}")
        released_prov = L.delivered(new["outputs"]["xlsx"]["file"]).replace("_SourceData.xlsx", "_SourceData_provenance.json")
        covered.add(released_prov)
        pb, pn = L.compare_provenance(PKG / released_prov, provs[0], TREE)
        rows.append(dict(generator=n, file=released_prov, status="equal apart from the inputs read" if not pb else "differs",
                         note=f"{len(pn)} input differences, each a declared public stand-in or a released file"))
        bad += [f"{released_prov}: {p}" for p in pb]
        notes += pn
    if set(names) >= {n for n, _ in L.GENERATORS}:
        released = {p.relative_to(PKG).as_posix() for d in ("figures/Main", "figures/Supplementary_Figures", "source_data") for p in (PKG / d).rglob("*") if p.is_file()}
        bad += [f"{f}: released but rebuilt by no generator" for f in sorted(released - covered)]
    return bad, dict(files=rows, input_differences=sorted(set(notes)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--only", nargs="*", help="generator names (figure1-5, ed01-ed10)")
    ap.add_argument("--external-result", type=Path, help="RESULT.json from the public ORD reproducer, outside this package; molecules.csv must be alongside it")
    ap.add_argument("--zhang", help="data/all_HTE_with_condition.csv of the pinned Zhang commit, plain or gzip (optional)")
    ap.add_argument("--journal", choices=("science", "natcatal"), help="must match JOURNAL_PROFILE.json; Science when absent")
    a = ap.parse_args()
    journal = L.journal()
    if a.journal is not None and a.journal != journal:
        ap.error(f"requested journal {a.journal!r} differs from the fixed package journal {journal!r}")
    names = set(a.only) if a.only else {n for n, _ in L.GENERATORS}
    unknown = names - {n for n, _ in L.GENERATORS}
    if unknown:
        raise SystemExit(f"unknown generators: {sorted(unknown)}")
    if "figure2" in names:
        if not a.external_result:
            ap.error("Figure 2 needs --external-result from code/External/replicate_public_screens.py; external results are not redistributed")
        a.external_result = a.external_result.resolve()
        if a.external_result.is_relative_to(PKG.resolve()):
            ap.error("external results must stay outside the P2 package")
        if not a.external_result.is_file() or not a.external_result.with_name("molecules.csv").is_file():
            ap.error("external RESULT.json or its molecules.csv is missing")
    if WORK.exists():                                             # read-only folders (as some synced drives mark them) are made writable
        shutil.rmtree(WORK, onexc=lambda func, path, exc: (os.chmod(path, 0o200 | 0o400), func(path)))
    for d in (TREE, OUT, WORK / "logs", WORK / "access_log", WORK / "tmp", WORK / "mplconfig"):
        d.mkdir(parents=True)
    os.environ["MPLCONFIGDIR"] = str(WORK / "mplconfig")         # before matplotlib is first imported: no font cache from elsewhere
    extra = [a.zhang] if a.zhang else []
    external_files = []
    if "figure2" in names:
        external_files = [str(a.external_result), str(a.external_result.with_name("molecules.csv"))]
    cfg = dict(read=allowed_read(extra), read_files=external_files, write=[str(WORK)], log=str(WORK / "access_log"))
    guard.install(cfg)                                            # this process too
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE")}
    env.update(P2_PUBLIC_GUARD=json.dumps(cfg), P2_RELEASE_MODE="public", PYTHONPATH=str(PKG / "code/release/guard"),
               P2_FIGURE_JOURNAL=journal,
               PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(WORK / "mplconfig"), TMP=str(WORK / "tmp"), TEMP=str(WORK / "tmp"),
               TMPDIR=str(WORK / "tmp"), PYTHONIOENCODING="utf-8")
    import matplotlib, numpy, pandas, scipy, sklearn, rdkit, openpyxl              # noqa: E401
    report = dict(package=PKG.name, started=time.strftime("%Y-%m-%d %H:%M:%S"), steps=[],
                  environment=dict(python=platform.python_version(), platform=platform.platform(), numpy=numpy.__version__,
                                   pandas=pandas.__version__, scipy=scipy.__version__, scikit_learn=sklearn.__version__,
                                   rdkit=rdkit.__version__, matplotlib=matplotlib.__version__, openpyxl=openpyxl.__version__),
                  inputs=dict(checksums_sha256=L.sha256(PKG / L.CHECKSUMS)), generators=sorted(names),
                  guard=dict(read=cfg["read"], read_files=cfg["read_files"], write=cfg["write"],
                             monitoring="Python file access logged; only this Python executable may start child processes; Python network connections refused"))
    if journal != "science":
        report["journal"] = journal
    def stage_inputs():
        L.stage(TREE, WORK)
        if "figure2" in names:
            spec = importlib.util.spec_from_file_location("external_plot_input", PKG / "code/Figures/external_figure_input.py")
            adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter)
            raw = adapter.encoded(adapter.public_projection(a.external_result, PKG))
            adapter.verify(raw)
            dest = TREE / "05_DATA/Results/external_replication/Figure2_plot_input.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            report["inputs"]["external_plot_sha256"] = adapter.EXPECTED_SHA256
            report["inputs"]["external_result_sha256"] = L.sha256(a.external_result)
            report["inputs"]["external_molecules_sha256"] = L.sha256(a.external_result.with_name("molecules.csv"))
        return [], dict(files=len(L.stage_map(WORK)), explicit_external_input="figure2" in names)
    ok = step("1 checksums of every released file", report, checksums)
    ok &= step("2 task construction: derived tables from the released records", report, task_construction)
    ok &= step("3 work tree: copies of released inputs and verified external display", report, stage_inputs)
    ok &= step("4 font list: the Arial files of the released figures first", report, font_list)
    ok &= step(f"5 figure generators in public mode ({len(names)})", report, lambda: generators(names, a.jobs, env))
    full = names == {n for n, _ in L.GENERATORS}
    if full:
        ok &= step(L.journal_formatter()("6 re-fit of Fig. S10 (fold registry blocked)"), report, lambda: refit(a.jobs, env))
        ok &= step("7 provenance flags", report, lambda: provenance(env, a.zhang))
        ok &= step("8 text statements and publication use of every measurement", report, lambda: claims_and_use(OUT / "figure3/Figure3_SourceData_provenance.json"))
    ok &= step("9 rebuilt outputs against the released ones", report, lambda: compare(names))
    if full:
        ok &= step("10 owner labels without the scored pairs; largest connected networks", report, lambda: nested(env))
        ok &= step("11 owner margins, equal-role null, SNAr class proxy and nesting", report, lambda: nullchk(env))
        ok &= step(L.journal_formatter()("12 tables S5 and S7, Notes 7, 10 and 15, and the counts"), report, lambda: sitables(env))
        ok &= step("13 values read from the full analysis run: Fig. 2A interval and likelihood owner, the Set A decision", report,
                   lambda: frozenreads(env))
    if any(name.startswith("figure") for name in names):
        ok &= step("14 independent main-figure values and actual series", report, lambda: figure_audits(names, env))
    report["result"] = "PASS" if ok else "FAIL"
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (WORK / "REBUILD_REPORT.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"rebuild: {report['result']}" + ("" if full else " (partial run)") + "; report in _rebuild/REBUILD_REPORT.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

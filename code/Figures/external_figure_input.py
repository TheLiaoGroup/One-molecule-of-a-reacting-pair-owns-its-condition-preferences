"""Project verified ORD reproduction results onto the approved Figure 2 display.

External rows stay outside the distributed P2 package. The package carries this
adapter and a hash, so external values cannot be silently substituted on rebuild.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "05_DATA/Results/external_replication/Figure2_plot_input.json"
ORIGINAL = ROOT / "05_DATA/Results/external_replication/RESULT.json"
EXPECTED_SHA256 = "c18d01ebcab0792ed45db6b521aec5fea11c213ca340693c799adb39363399ae"


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def project(value):
    datasets = value["datasets"]
    ch, suzuki = datasets["CH_arylation"], datasets["Suzuki"]
    roles = ("halide_consistency", "partner_consistency")
    result = dict(schema="p2_v6_external_plot_input_v1", datasets=dict(
        CH_arylation=dict(main=dict(estimates={key: ch["main"]["estimates"][key] for key in roles}),
                         consistency_bootstrap=dict(difference_ci95=ch["consistency_bootstrap"]["difference_ci95"])),
        Suzuki=dict(point={key: suzuki["point"][key] for key in roles},
                    bootstrap=dict(difference_ci95=suzuki["bootstrap"]["difference_ci95"]))),
        benchmarks={name: {role: value["benchmarks"][name][role] for role in ("halide", "partner")}
                    for name in ("BH_ligand", "Suzuki")}, molecules={})
    for name in ("CH_arylation", "Suzuki"):
        rows = [{key: row[key] for key in ("role", "status", "molecule", "consistency")}
                for row in value["molecules"][name] if row["status"] == "eligible"]
        result["molecules"][name] = sorted(rows, key=lambda row: (row["role"], row["molecule"]))
    # Stable display serialization removes sub-ulp arithmetic-order differences
    # between the original analysis and its independently verified reproduction.
    # Full-precision scientific results remain untouched in the source records.
    def display(value):
        if isinstance(value, float):
            shown = round(value, 15)
            assert abs(shown - value) < 1e-12
            return shown
        if isinstance(value, dict):
            return {key: display(item) for key, item in value.items()}
        if isinstance(value, list):
            return [display(item) for item in value]
        return value
    return display(result)


def public_projection(result_file, package):
    """Read only outputs of replicate_public_screens.py and released P2 medians."""
    import openpyxl
    result_file, package = Path(result_file), Path(package)
    public = json.loads(result_file.read_text(encoding="utf-8"))
    ch, suzuki = public["datasets"]["CH_arylation"], public["datasets"]["Suzuki"]
    with (result_file.parent / "molecules.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["consistency"] = float(row["consistency"]) if row["consistency"] else None
    workbook = openpyxl.load_workbook(package / "figure_inputs/v5_reference_tables/preference.xlsx", read_only=True, data_only=True)
    try:
        cells = workbook["C_group_medians"].values
        header = next(cells)
        medians = [dict(zip(header, row)) for row in cells]
    finally:
        workbook.close()
    medians = [row for row in medians if row["screen"] == "Buchwald-Hartwig|1"]
    assert len(medians) == 2 and {row["role"] for row in medians} == {1, 2}
    bh = {row["role"]: row["median"] for row in medians}
    reference = suzuki["training_benchmark"]["point"]
    value = dict(datasets=dict(CH_arylation=dict(main=ch["main"],
                    consistency_bootstrap=ch["main"]["consistency_interval"]), Suzuki=suzuki),
                 molecules={name: [row for row in rows if row["dataset"] == name
                                   and (name != "CH_arylation" or row["variant"] == "main")]
                            for name in ("CH_arylation", "Suzuki")},
                 benchmarks=dict(BH_ligand=dict(halide=bh[2], partner=bh[1]),
                                 Suzuki=dict(halide=reference["halide_consistency"], partner=reference["partner_consistency"])))
    return project(value)


def verify(raw):
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256, "External plotted identities or values differ from the approved frozen result"
    value = json.loads(raw)
    assert encoded(value) == raw, "External display JSON is not canonical"
    return value


def load_plot_input(path=None):
    path = Path(path) if path is not None else CANONICAL
    value = verify(path.read_bytes())
    return value, path, EXPECTED_SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-internal", action="store_true")
    parser.add_argument("--external-result", type=Path)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.freeze_internal:
        assert not CANONICAL.exists(), "Preserve the first frozen display input"
        raw = encoded(project(json.loads(ORIGINAL.read_text(encoding="utf-8"))))
        CANONICAL.write_bytes(raw)
        print(hashlib.sha256(raw).hexdigest())
        return
    assert args.external_result and args.package and args.out
    assert not args.out.exists(), "Do not overwrite previous external verification evidence"
    raw = encoded(public_projection(args.external_result, args.package))
    verify(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(raw)
    print(f"Verified external Figure 2 input: {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()

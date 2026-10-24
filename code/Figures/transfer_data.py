"""Retained scientific extraction, independent of figure rendering."""
import json
import numpy as np
import pandas as pd
from figure_common import ARRAY_LABELS, ROOT, array_label

KS = (1, 2, 3, 4, 6)


OWNER = "#30343B"


PARTNER = "#8A9098"


ROLE_PALETTE = {"halide": "#0072B2", "amine": "#D55E00", "acid": "#CC79A7", "other": "#009E73"}


FILES = {
    "A": "05_DATA/Results/reserve_validation/record_transfer_losses.csv",
    "B": "05_DATA/Results/cross_library/target_losses.csv",
    "k2": "05_DATA/Results/directional/phase2_setAB_k2.csv",
    "internal": "05_DATA/Results/directional/phase3_transport.csv",
    "depth": "05_DATA/Results/primary/anatomy_depth.csv",
    "depth_folds": "05_DATA/Analysis/figure_display/Fig4a_folds.csv",
    "roles": "05_DATA/Results/display_inputs/Anatomy_arrays.csv",
    "summary_A": "05_DATA/Results/reserve_validation/reserve_validation.json",
    "summary_B": "05_DATA/Results/cross_library/cross_library.json",
    "summary_k2": "05_DATA/Results/directional/phase2_summary.json",
    "summary_internal": "05_DATA/Results/directional/phase3_summary.json",
}


def read_json(key):
    return json.loads((ROOT / FILES[key]).read_text(encoding="utf-8"))


def fractions(frame, owner_col, partner_col):
    assert len(frame) and frame.sst.sum() > 0
    return (float(1 - frame[owner_col].sum() / frame.sst.sum()),
            float(1 - frame[partner_col].sum() / frame.sst.sum()))


def role_key(role):
    """Match Figure 2's colour classes using the recorded owner role."""
    role = str(role).replace("\N{EN DASH}", "-")
    if "halide" in role:
        return "halide"
    if role in ("N-H partner", "N-H nucleophile", "amine"):
        return "amine"
    if role == "carboxylic acid":
        return "acid"
    return "other"


def point_source_tables(k2, internal, depth):
    targets = k2.rename(columns={
        "electrophile": "owner_molecule", "sse_e": "sse_owner", "sse_n": "sse_partner",
        "sse_e_k2": "sse_owner_two_partners", "sse_n_k2": "sse_partner_two_partners",
        "subsets_e": "owner_subsets", "subsets_n": "partner_subsets",
        "history_e": "owner_history_pairs", "history_n": "partner_history_pairs",
    }).copy()
    targets["array"] = targets.screen.map(lambda key: array_label(key, mathtext=False))
    targets = targets[["set", "screen", "array", "subpair", "owner_molecule", "sse_owner", "sse_partner",
                       "sst", "sse_owner_two_partners", "owner_subsets", "owner_history_pairs",
                       "sse_partner_two_partners", "partner_subsets", "partner_history_pairs"]]
    transport = internal.copy()
    transport["array"] = transport.screen.map(lambda key: array_label(key, mathtext=False))
    transport = transport[["screen", "array", "subpair", "sst", "sse_owner", "sse_partner",
                           "subsets_owner", "subsets_partner"]]

    # The released folds retain the sorted pair universe used by anatomy_depth.csv.
    folds = pd.read_csv(ROOT / FILES["depth_folds"])
    assert not folds.duplicated(["screen", "subpair"]).any()
    identities, molecules = {}, {}
    for screen, group in folds.groupby("screen"):
        group = group.sort_values("subpair", kind="stable")
        identities[screen] = list(zip(group.subpair, group.fold))
        molecules[screen] = {role: sorted({pair.split(" >> ")[role - 1] for pair in group.subpair})
                             for role in (1, 2)}
    records = []
    for row in depth.itertuples():
        pair, fold = identities[row.screen][row.pair]
        molecule = pair.split(" >> ")[row.role - 1]
        assert molecules[row.screen][row.role][row.molecule] == molecule
        records.append(dict(screen=row.screen, array=array_label(row.screen, mathtext=False),
                            role=row.role, side=row.side, target_pair=pair, molecule=molecule, fold=int(fold),
                            sst=row.sst, **{f"sse_{k}": getattr(row, f"sse_{k}") for k in KS},
                            pair_index=row.pair, molecule_index=row.molecule, common_target=row.common_target))
    assert len(targets) == 844 and len(transport) == 908 and len(records) == 1247
    assert targets[["sse_owner_two_partners", "sse_partner_two_partners"]].notna().all().all()
    return targets, transport, pd.DataFrame(records)


def load_data():
    loss = {name: pd.read_csv(ROOT / FILES[name]) for name in ("A", "B")}
    k2 = pd.read_csv(ROOT / FILES["k2"])
    internal = pd.read_csv(ROOT / FILES["internal"])
    depth = pd.read_csv(ROOT / FILES["depth"])
    owner_table = pd.read_csv(ROOT / FILES["roles"]).set_index("screen")
    owner_roles = owner_table.owner_role
    owner_colours = owner_table.owner.map(role_key).map(ROLE_PALETTE).to_dict()
    depth["side"] = ["owner" if row.role == owner_roles[row.screen] else "partner"
                     for row in depth.itertuples()]
    assert not depth.duplicated(["screen", "pair", "role"]).any()
    common = depth.groupby(["screen", "pair"]).side.nunique()
    common = common[common == 2].index
    depth["common_target"] = pd.MultiIndex.from_frame(depth[["screen", "pair"]]).isin(common)
    shared = depth[depth.common_target].copy()
    assert len(common) == 310 and len(shared) == 620
    expected = {"Acylation|1": 236, "Suzuki|1": 66, "Buchwald-Hartwig|1": 4,
                "Kumada|1": 3, "Sonogashira|1": 1}
    assert shared.groupby("screen").pair.nunique().to_dict() == expected
    assert shared.groupby(["screen", "pair"]).sst.nunique().eq(1).all()

    b_rows, c_rows = [], []
    for name, frame in loss.items():
        summary = read_json("summary_" + name)
        if name == "A":
            summary = summary["e2"]
        o, p = fractions(frame, "sse_e", "sse_n")
        assert len(frame) == summary["targets"]
        assert frame.screen.nunique() == summary["arrays"]
        assert np.allclose([o, p], [summary["r2_electrophile"], summary["r2_partner"]], atol=1e-12)
        b_rows.append(dict(set="Set " + name, row_type="pooled", screen="all", owner=o, partner=p,
                           targets=len(frame), arrays=frame.screen.nunique()))
        for screen, group in frame.groupby("screen"):
            go, gp = fractions(group, "sse_e", "sse_n")
            b_rows.append(dict(set="Set " + name, row_type="array", screen=screen, owner=go, partner=gp,
                               targets=len(group), arrays=1))
        group = k2[k2["set"] == "Set " + name]
        assert set(zip(frame.screen, frame.subpair)) == set(zip(group.screen, group.subpair))
        co, cp = fractions(group, "sse_e_k2", "sse_n_k2")
        reference = read_json("summary_k2")["7.1"]["Set " + name]
        assert np.allclose([co, cp], [reference["r2_owner_k2"], reference["r2_partner_k2"]], atol=1e-12)
        c_rows.append(dict(cohort="Out-of-array: Set " + name, set="Set " + name, owner=co, partner=cp,
                           targets=len(group), arrays=group.screen.nunique(), source_file=FILES["k2"],
                           source_columns="sst; sse_e_k2; sse_n_k2"))
    io, ip = fractions(internal, "sse_owner", "sse_partner")
    reference = read_json("summary_internal")["7.2"]
    assert len(internal) == reference["targets"] == 908
    assert np.allclose([io, ip], [reference["r2_owner"], reference["r2_partner"]], atol=1e-12)
    c_rows.append(dict(cohort="Within-array held-out", set="Analysis arrays", owner=io, partner=ip,
                       targets=len(internal), arrays=internal.screen.nunique(), source_file=FILES["internal"],
                       source_columns="sst; sse_owner; sse_partner"))
    d_rows = []
    for side, group in shared.groupby("side"):
        for k in KS:
            d_rows.append(dict(side=side, partners=k, explained_pct=float(100 * (1 - group[f"sse_{k}"].sum() / group.sst.sum())),
                               targets=group.pair.count(), arrays=group.screen.nunique(),
                               sum_sse=float(group[f"sse_{k}"].sum()), sum_sst=float(group.sst.sum())))

    targets, transport, depth_records = point_source_tables(k2, internal, depth)
    b, c = pd.DataFrame(b_rows), pd.DataFrame(c_rows)
    for frame in (b, c):
        frame["owner_display_pct"] = 100 * frame.owner
        frame["partner_display_pct"] = 100 * frame.partner
        frame["owner_minus_partner_display_points"] = 100 * (frame.owner - frame.partner)
    b["owner_molecular_role"] = b.screen.map(owner_table.owner).fillna("mixed roles")
    b["owner_colour"] = b.screen.map(owner_colours).fillna(OWNER)
    b["partner_colour"] = PARTNER
    b["array"] = b.screen.map(ARRAY_LABELS).fillna("All arrays")
    return dict(loss=loss, k2=k2, internal=internal, depth=depth, shared=shared,
                b=b, c=c, d=pd.DataFrame(d_rows), owner_colours=owner_colours,
                original_targets=targets, original_depth=depth_records, original_internal=transport)

"""Draft Figure 4: prediction for new pairs and growth of molecular history."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_figure_common as c


import hashlib
import numpy as np
import pandas as pd
import transfer_data as old
from figure_common import array_label
RULE = c.ROOT / "07_METHODS_AND_PROVENANCE/frozen_analysis_plans/figure_selection_v3/RULE.md"
RULE_HASH = "1eef02cbdb55db159135883820e4a83e353e7da0f057a2fd258f96082143c02d"

def selection(data):
    actual = hashlib.sha256(RULE.read_bytes()).hexdigest()
    assert actual == RULE_HASH
    counts = data["shared"].groupby("screen").pair.nunique().sort_index()
    selected = counts[counts >= 20].index.tolist()
    assert selected == ["Acylation|1", "Suzuki|1"] and int(counts.sum()) == 310
    curves = []
    for (screen, side), group in data["shared"].groupby(["screen", "side"]):
        if screen in selected:
            for k in old.KS:
                curves.append(dict(screen=screen, array=array_label(screen, mathtext=False), side=side,
                                   partners=k, common_targets=int(group.pair.nunique()),
                                   explained_pct=100 * (1 - group[f"sse_{k}"].sum() / group.sst.sum()),
                                   sum_sse=float(group[f"sse_{k}"].sum()), sum_sst=float(group.sst.sum())))
    record = dict(rule_sha256=actual, threshold_common_targets=20, common_targets=310,
                  aggregation="100 * (1 - sum(SSE_k) / sum(SST)); retained errors only; no refitting",
                  array_counts={key: int(value) for key, value in counts.items()},
                  selected_arrays=[dict(screen=key, name=array_label(key, mathtext=False), targets=int(counts[key])) for key in selected],
                  pooled_arrays=counts.index.tolist(), omitted_thin_arrays=counts[counts < 20].index.tolist())
    return record, pd.DataFrame(curves)


HEIGHT = 198.0
TAGS = {"A": "New pairs", "B": "Equal history depth", "C": "History growth"}


def role_color(role):
    role = str(role).lower()
    if "halide" in role or "electrophile" in role:
        return c.BLUE
    if "amine" in role or "n\u2013h" in role or "n-h" in role:
        return c.ORANGE
    return c.DARK


def printed_numbers(tables):
    ledger = tables["Printed_numbers"]
    pooled = ((ledger.panel == "A") & (ledger.row_type == "pooled")
              & ledger.side.isin(["owner", "partner"]))
    endpoint = ((ledger.panel == "C") & (ledger.history_partners == 6)
                & ledger.source_selection.isin(["common_target=True;side=owner",
                                                 "common_target=True;side=partner"]))
    shown = ledger[pooled | endpoint].copy()
    assert len(shown) == 6 and int(pooled.sum()) == 4 and int(endpoint.sum()) == 2
    shown["display"] = [f"{row.value:.1f}%" if row.panel == "C" else f"{row.value:.1f}"
                        for row in shown.itertuples()]
    return shown


def draw(tables):
    points = tables["A_array_and_pooled"]
    equal = tables["B_equal_depth"]
    pooled_curves = tables["C_pooled_curves"]
    thin_curves = tables["C_selected_array_curves"]
    assert equal["set"].tolist() == ["Set A", "Set B", "Analysis arrays"]
    assert set(thin_curves.screen) == {"Acylation|1", "Suzuki|1"}
    assert set(pooled_curves.targets) == {310}
    fig, canvas = c.new_figure(HEIGHT)
    c.panel(canvas, "A", 1, 3, TAGS["A"])
    c.panel(canvas, "B", 110, 3, TAGS["B"])
    c.panel(canvas, "C", 1, 113, TAGS["C"])

    axis = c.axes_mm(fig, 18, 17, 84, 79)
    axis.set(xlim=(-20, 29), ylim=(-8, 32), xticks=[-20, 0, 20],
             yticks=[0, 10, 20, 30])
    axis.axline((0, 0), slope=1, color=c.LIGHT, linewidth=.5,
                linestyle=(0, (3, 3)), zorder=0)
    axis.axhline(0, color=c.LIGHT, linewidth=.5, zorder=0)
    axis.axvline(0, color=c.LIGHT, linewidth=.5, zorder=0)
    for row in points[points.row_type == "array"].itertuples():
        color = role_color(row.owner_molecular_role)
        marker = "o" if row.set == "Set A" else "s"
        axis.plot(row.partner_display_pct, row.owner_display_pct,
                  marker=marker, markersize=4, color=color, linewidth=0,
                  markeredgewidth=.5, gid=f"A|{row.set}|{row.screen}")
    summary = []
    for row in points[points.row_type == "pooled"].itertuples():
        marker = "o" if row.set == "Set A" else "s"
        axis.plot(row.partner_display_pct, row.owner_display_pct,
                  marker=marker, markersize=6.5, color=c.DARK,
                  markeredgecolor="white", markeredgewidth=.7, linewidth=0,
                  zorder=4, gid="A|pooled|" + row.set)
        label = f"{row.set}\n{row.owner_display_pct:.1f} / {row.partner_display_pct:.1f}"
        offset = (15, 23) if row.set == "Set A" else (8, 7)
        axis.annotate(label, (row.partner_display_pct, row.owner_display_pct),
                      xytext=offset, textcoords="offset points", fontsize=8,
                      va="bottom",
                      linespacing=1.0, arrowprops=dict(arrowstyle="-", color=c.DARK, lw=.5),
                      gid="A|summary|" + row.set)
        summary.append(dict(set=row.set, owner=row.owner_display_pct,
                            partner=row.partner_display_pct))
    axis.set_xlabel("Partner: explained preference (%)", fontsize=8, labelpad=3)
    axis.set_ylabel("Owner: explained preference (%)", fontsize=8, labelpad=3)

    axis = c.axes_mm(fig, 139, 24, 36, 63)
    axis.set(xlim=(0, 16), ylim=(3.5, .5), xticks=[0, 8, 16], yticks=[])
    axis.spines["left"].set_visible(False)
    for position, row in enumerate(equal.itertuples(), 1):
        axis.plot([row.partner_display_pct, row.owner_display_pct],
                  [position, position], color=c.LIGHT, linewidth=.7,
                  gid=f"B|connector|{row.set}")
        axis.plot(row.partner_display_pct, position, "o", markersize=4,
                  markerfacecolor="white", markeredgecolor=c.DARK,
                  markeredgewidth=.5, gid=f"B|partner|{row.set}")
        axis.plot(row.owner_display_pct, position, "o", markersize=4,
                  markerfacecolor=c.DARK, markeredgecolor=c.DARK,
                  markeredgewidth=.5, gid=f"B|owner|{row.set}")
        y = 24 + (position - .5) * 21
        canvas.text(110, y, ("Set A", "Set B", "Within-screen")[position - 1],
                    fontsize=8, va="center")
    first = equal.iloc[0]
    axis.annotate("Partner", (first.partner_display_pct, 1), xytext=(-2, 8),
                  textcoords="offset points", fontsize=8, ha="right")
    axis.annotate("Owner", (first.owner_display_pct, 1), xytext=(0, 8),
                  textcoords="offset points", fontsize=8, ha="center")
    canvas.text(153, 104, "Explained preference (%)", ha="center",
                va="center", fontsize=8, gid="B|axis_label")

    axis = c.axes_mm(fig, 18, 124, 151, 58)
    axis.set(xlim=(.8, 7.65), ylim=(-7, 34), xticks=[1, 2, 3, 4, 6],
             yticks=[0, 10, 20, 30])
    axis.axhline(0, color=c.LIGHT, linewidth=.5, zorder=0)
    for (screen, side), group in thin_curves.groupby(["screen", "side"], sort=False):
        group = group.sort_values("partners", kind="stable")
        color = c.ORANGE if screen == "Acylation|1" else c.BLUE
        axis.plot(group.partners, group.explained_pct, color=color,
                  linewidth=.6, linestyle="-" if side == "owner" else (0, (3, 2)),
                  zorder=1, gid=f"C|thin|{screen}|{side}")
        last = group.iloc[-1]
        label = "Amide coupling" if screen == "Acylation|1" else "Suzuki"
        shift = {("Acylation|1", "owner"): 10,
                 ("Acylation|1", "partner"): 8,
                 ("Suzuki|1", "owner"): 0,
                 ("Suzuki|1", "partner"): 0}[(screen, side)]
        if side == "owner":
            axis.annotate(label, (last.partners, last.explained_pct),
                          xytext=(8, shift), textcoords="offset points", fontsize=8,
                          va="center", gid=f"C|label|{screen}|{side}")
    endpoints = []
    for side, group in pooled_curves.groupby("side", sort=False):
        group = group.sort_values("partners", kind="stable")
        color = c.DARK if side == "owner" else c.LIGHT
        axis.plot(group.partners, group.explained_pct, color=color,
                  linewidth=1.25, linestyle="-" if side == "owner" else (0, (3, 2)),
                  marker="o", markersize=3, markeredgewidth=.5,
                  markerfacecolor=color if side == "owner" else "white",
                  markeredgecolor=color, zorder=3, gid="C|pooled|" + side)
        last = group.iloc[-1]
        label = f"{side.capitalize()} {last.explained_pct:.1f}%"
        axis.annotate(label, (last.partners, last.explained_pct), xytext=(8, 0),
                      textcoords="offset points", fontsize=8, va="center",
                      gid="C|summary|" + side)
        endpoints.append(dict(side=side, partners=int(last.partners),
                              explained_pct=float(last.explained_pct)))
    axis.set_xlabel("Historical partners", fontsize=8, labelpad=3)
    axis.set_ylabel("Profile explained (%)", fontsize=8, labelpad=3)
    canvas.text(173, 116, "dashed: partner", ha="right", fontsize=8,
                gid="C|dashed_legend")
    return fig, dict(panel_tags=TAGS, pooled_summaries=summary,
                     pooled_endpoints=endpoints,
                     selected_arrays=["Acylation|1", "Suzuki|1"], common_targets=310)


def main():
    tables = c.read_tables(5)
    record, curves = selection(old.load_data())
    columns = ["screen", "side", "partners", "common_targets", "explained_pct", "sum_sse", "sum_sst"]
    actual = curves[columns].sort_values(columns[:3]).reset_index(drop=True)
    expected = tables["C_selected_array_curves"][columns].sort_values(columns[:3]).reset_index(drop=True)
    assert actual[columns[:2]].equals(expected[columns[:2]])
    assert np.allclose(actual[columns[2:]].astype(float), expected[columns[2:]].astype(float), atol=1e-12, rtol=1e-12)
    c.write_json(4, "figure4_selection.json", record)
    tables["V5_printed_numbers"] = printed_numbers(tables)
    with c.style():
        fig, observation = draw(tables)
        return c.save(fig, 4, tables, extra=observation)


if __name__ == "__main__":
    c.cli(main, 4)

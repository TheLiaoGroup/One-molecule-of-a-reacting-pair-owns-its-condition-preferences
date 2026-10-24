"""Draft Figure 3: the same molecules in different reaction contexts."""
from pathlib import Path
import sys
import textwrap
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import main_figure_common as c


ORDER = ("M4", "M8", "M1", "M2", "M3", "M5", "M7", "M6")
HEIGHT = 214.0
TAGS = {"A": "Aromatic amines", "B": "Molecular contexts"}


def summary_value(tables, field):
    ledger = tables["Printed_numbers"]
    match = ledger[ledger.source_field == field]
    assert len(match) == 1, field
    return float(match.iloc[0].value)


def molecule_label(value):
    return "\n".join(
        part
        for line in str(value).splitlines()
        for part in textwrap.wrap(line, width=28, break_long_words=False,
                                  break_on_hyphens=True)
    )


def printed_numbers(tables):
    return pd.DataFrame(dict(panel=panel, quantity="axis tick", value=value,
                               display=f"{value:.1f}" if value else "0.0",
                               source_file="display coordinate", source_field="axis tick",
                               source_selection="condition-preference fraction")
                          for panel, values in (("A", (-.4, 0., .4)), ("B", (-.4, 0., .4, .8)))
                          for value in values)


def draw(tables):
    amines = tables["A_eight_amines"]
    molecules = tables["B_23_molecules"]
    assert tuple(amines.plot_label) == ORDER
    assert molecules.plot_row.tolist() == list(range(1, 24))
    fig, canvas = c.new_figure(HEIGHT)
    c.panel(canvas, "A", 1, 3, TAGS["A"])
    c.panel(canvas, "B", 93, 3, TAGS["B"])

    for index, row in enumerate(amines.itertuples()):
        column, grid_row = index % 2, index // 2
        left, top = 1 + 44 * column, 14 + 47 * grid_row
        path = c.ROOT / f"02_FIGURES/ChemDraw/Fig2_b_{row.plot_label}.svg"
        width, height = c.structure_size(path)
        c.structure(canvas, path, left + 26 - width / 2,
                    top + (23 - height) / 2, f"A|structure|{row.plot_label}")
        axis = c.axes_mm(fig, left + 12, top + 25, 29, 16)
        axis.set(xlim=(-.1, 2.1), ylim=(-.48, .70), xticks=[0, 1, 2],
                 yticks=[-.4, 0, .4])
        axis.tick_params(axis="both", pad=1)
        if grid_row == 3:
            axis.set_xticklabels(["Amide", "Pd", "Pd/Cu/Ni"])
        else:
            axis.set_xticklabels([])
        if column:
            axis.set_yticklabels([])
        axis.axhline(0, color=c.LIGHT, linewidth=.5, zorder=0)
        color = c.LIGHT if row.plot_label in ("M3", "M8") else c.ORANGE
        axis.plot([0, 1, 2], [row.T_amide, row.T_BH1, row.T_BH2],
                  color=color, linewidth=.7, marker="o", markersize=3,
                  markeredgewidth=.5, gid=f"A|path|{row.plot_label}")
    canvas.text(2, 105, "Preference across partners", rotation=90,
                ha="center", va="center", fontsize=8)
    gains = [summary_value(tables, "C." + field) for field in
             ("mean_amide_minus_BH1", "mean_amide_minus_BH2")]

    mean_owner = summary_value(tables, "descriptives.T_O_mean")
    mean_partner = summary_value(tables, "descriptives.T_N_mean")
    canvas.plot(123, 19, "o", markersize=3.5, color=c.DARK,
                markeredgewidth=.5)
    canvas.text(126, 19, "Owner", va="center", fontsize=8, gid="B|legend_owner")
    canvas.plot(149, 19, "o", markersize=3.5, markerfacecolor="white",
                markeredgecolor=c.DARK, markeredgewidth=.5)
    canvas.text(152, 19, "Non-owner", va="center", fontsize=8, gid="B|legend_nonowner")
    axis = c.axes_mm(fig, 141, 25, 37, 172.5)
    axis.set(xlim=(-.52, .88), ylim=(23.5, .5), yticks=[],
             xticks=[-.4, 0, .4, .8])
    axis.spines["left"].set_visible(False)
    axis.axvline(0, color=c.LIGHT, linewidth=.5, zorder=0)
    for row in molecules.itertuples():
        position = row.plot_row
        y = 25 + (position - .5) * 7.5
        canvas.text(94, y, molecule_label(row.name_display), fontsize=8,
                    va="center", linespacing=1.0, gid=f"B|name|{position}")
        color = c.ORANGE if row.stratum == "amine" else c.DARK
        axis.plot([row.T_N, row.T_O], [position, position], color=c.LIGHT,
                  linewidth=.7, gid=f"B|connector|{position}")
        axis.plot(row.T_N, position, "o", markersize=3.5,
                  markerfacecolor="white", markeredgecolor=color,
                  markeredgewidth=.5, gid=f"B|nonowner|{position}")
        axis.plot(row.T_O, position, "o", markersize=3.5,
                  markerfacecolor=color, markeredgecolor=color,
                  markeredgewidth=.5, gid=f"B|owner|{position}")
    canvas.text(142, 208, "Preference across partners", fontsize=8,
                va="center", ha="center")
    return fig, dict(panel_tags=TAGS, frozen_amine_order=list(ORDER),
                     displayed_summaries={"A": gains,
                                          "B": [mean_owner, mean_partner]})


def main():
    tables = c.read_tables(4)
    tables["V5_printed_numbers"] = printed_numbers(tables)
    with c.style():
        fig, observation = draw(tables)
        return c.save(fig, 3, tables, extra=observation)


if __name__ == "__main__":
    c.cli(main, 3)

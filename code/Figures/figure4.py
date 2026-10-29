"""Figure 4 and shared approved v6 transfer-drawing helpers."""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import itertools
import json
import re
import shutil
import textwrap
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.text import Text
import numpy as np
import pandas as pd
import pymupdf

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "06_CODE/Figures"))
import main_figure_common as C
import v6_core as c
import structure_art
from figure3_selection import load_art
from figure1_data import public_units, array_patterns

PTMM = 25.4 / 72
BLUE, ORANGE, DARK, LIGHT = C.BLUE, C.ORANGE, C.DARK, C.LIGHT
INPUTS = set()
STRUCTURES = []
FROZEN_HALIDE = "Cc1cc(F)ccc1C#N"
FROZEN_NH = "c1cc2[nH]ccc2cn1"
ORDER = ("M4", "M8", "M1", "M2", "M3", "M5", "M7", "M6")
DISPLAY = ROOT / "05_DATA/Analysis/figure_display/v6_figure4"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_tables(number):
    path = C.reference_path(number)
    INPUTS.add(path)
    INPUTS.add(C.REFERENCE / "MANIFEST.json")
    return c.tables({3:2}.get(number,number))


def jsave(path, data):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        return value
    return c.write_json(Path(path).name,clean(data))


def style():
    return plt.rc_context({
        "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
        "axes.titlesize": 7, "axes.linewidth": .5, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": .5, "ytick.major.width": .5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "lines.linewidth": .7, "pdf.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "P2-v6-figure45", "mathtext.fontset": "custom",
        "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
        "text.color": "black", "axes.grid": False,
    })


def figure(height):
    fig = plt.figure(figsize=(180 / 25.4, height / 25.4), dpi=180)
    can = axmm(fig, 0, 0, 180, height)
    can.set(xlim=(0, 180), ylim=(height, 0))
    can.axis("off")
    return fig, can


def axmm(fig, x, y, w, h):
    return C.axes_mm(fig, x, y, w, h)


def panel(can, letter, x, y, label):
    can.text(x, y, letter, fontsize=8, fontweight="bold", va="center", gid="panel_" + letter)
    if label.startswith(r"S$_{\mathrm{N}}$Ar"):
        # Native text runs keep the upright subscript above the 5 pt minimum.
        for dx,dy,value,size in ((5,.85,"S",7),(6.65,1.4,"N",5.25),(8.0,.85,"Ar",7),
                                 (10.35,.85,label[len(r"S$_{\mathrm{N}}$Ar"):],7)):
            can.text(x+dx,y+dy,value,fontsize=size,va="baseline",gid="panel_title_"+letter)
    else:
        can.text(x + 5, y, label, fontsize=7, va="center", gid="panel_title_" + letter)


def structure(can, path, cx, cy, gid):
    path = Path(path)
    art = load_art(path)
    w, h = structure_art.size_mm(art, 7)
    structure_art.draw(can, art, cx - w / 2, cy - h / 2, 7, gid)
    for patch in can.patches:
        if (patch.get_gid() or "").startswith(gid + "|bond"):
            patch.set_edgecolor("black")
            patch.set_linewidth(.08)
    INPUTS.add(path)
    cdxml = path.with_suffix(".cdxml")
    assert cdxml.exists(), cdxml
    INPUTS.add(cdxml)
    STRUCTURES.append(dict(gid=gid, svg=str(path), cdxml=str(cdxml),
                           width_mm=float(w), height_mm=float(h), center_mm=[cx, cy],
                           master_bond_pt=14.4, master_atom_label_pt=10,
                           common_print_factor=.7, atom_label_pt=7, subscript_pt=5.25,
                           printed_bond_pt=10.08, added_polygon_outline_pt=.08,
                           effective_printed_bond_width_pt=.50))


def summary(tables, field):
    rows = tables["Printed_numbers"]
    match = rows[rows.source_field == field]
    assert len(match) == 1, field
    return float(match.iloc[0].value)


def snar_data():
    INPUTS.add(ROOT / "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz")
    INPUTS.add(ROOT / "06_CODE/Figures/array_shares_data.py")
    INPUTS.add(ROOT / "06_CODE/Figures/figure1_data.py")
    INPUTS.add(DISPLAY / "SNAr_display_rule.md")
    measurements, _ = public_units()
    ap = array_patterns(measurements, "SNAr|1")
    conditions = read_tables(3)["Fig3B_conditions"].copy()
    conditions = conditions.sort_values(["base_or_acid", "Standardized Name", "condition_id"], kind="stable")
    conds = conditions.condition_id.tolist()
    assert len(conds) == 47 and set(conds) == set(ap["conds"])
    columns = [ap["conds"].index(k) for k in conds]
    candidates = sorted([(int((ap["a"] == s).sum()), str(s)) for s in set(ap["a"])], key=lambda x: (-x[0], x[1]))
    focal_nh = candidates[0][1]
    assert focal_nh == "CC1CCCNC1" and candidates[0][0] == 8
    assert int((ap["a"] == FROZEN_NH).sum()) == 1
    blocks, cells, identities = [], [], []
    for key, focal, fixed, changed in (("halide", FROZEN_HALIDE, "b", "a"), ("NH", focal_nh, "a", "b")):
        ix = sorted(np.where(ap[fixed] == focal)[0], key=lambda i: str(ap[changed][i]))
        z = ap["E"][ix][:, columns]
        for row, i in enumerate(ix, 1):
            row_id = ("H" if key == "halide" else "N") + str(row)
            identities.append(dict(block=key, row_id=row_id, fixed_smiles=focal,
                                   partner_smiles=str(ap[changed][i]), source_subpair=ap["pairs"][i],
                                   measured_conditions=int(np.isfinite(z[row-1]).sum())))
            for j, condition in enumerate(conds):
                cells.append(dict(block=key, row_id=row_id, screen="SNAr|1", source_subpair=ap["pairs"][i],
                                  condition_id=condition, display_column=j+1,
                                  yield_fraction=float(ap["Y"][i, columns[j]]),
                                  preference_pp=float(z[row-1, j]), clipped_pp=float(np.clip(z[row-1, j], -30, 30))))
        blocks.append(dict(key=key, focal=focal, rows=ix, z=z,
                           row_ids=[("H" if key == "halide" else "N") + str(k+1) for k in range(len(ix))]))
    arrays = read_tables(3)["Fig3A_arrays"]
    shares = arrays[arrays.screen == "SNAr|1"].iloc[0]
    c.write_csv("Figure4c_fingerprint_cells.csv",pd.DataFrame(cells))
    c.write_csv("Figure4c_fingerprint_row_key.csv",pd.DataFrame(identities))
    c.write_csv("Figure4c_condition_order.csv",conditions)
    jsave(OUT / "Figure4c_selection.json", dict(rule_sha256=digest(DISPLAY / "SNAr_display_rule.md"),
          fixed_halide=FROZEN_HALIDE, fixed_NH=focal_nh,
          original_illustrated_NH=FROZEN_NH, original_NH_product_forming_partners=1,
          candidate_counts=[dict(smiles=s, product_forming_partners=n) for n,s in candidates],
          blocks=identities, condition_order=conds, two_independent_focal_experiments=True,
          screen_role_shares=dict(halide=float(shares.owner_share), NH=float(shares.partner_share))))
    return blocks, conditions, shares


def draw4():
    tables = read_tables(4)
    amines, molecules = tables["A_eight_amines"], tables["B_23_molecules"]
    assert tuple(amines.plot_label) == ORDER
    assert int((molecules.T_O > molecules.T_N).sum()) == 17
    means = [summary(tables, "descriptives.T_O_mean"), summary(tables, "descriptives.T_N_mean")]
    assert np.allclose(means, [molecules.T_O.mean(), molecules.T_N.mean()])
    blocks, conditions, shares = snar_data()
    fig, can = figure(215)
    panel(can, "a", 1, 3, "Amide > C\u2013N mean: 6/8")
    can.text(53, 45.5, "Grey: two reverse cases", fontsize=6, color=DARK, va="center",
             gid="4a_reverse_legend")
    for i, row in enumerate(amines.itertuples()):
        col, gridrow = i % 2, i // 2
        x, top = 12 + 44 * col, 12 + 39 * gridrow
        structure(can, ROOT / f"02_FIGURES/ChemDraw/Fig2_b_{row.plot_label}.svg", x + 14.5, top + 8.5,
                  "4a_structure_" + row.plot_label)
        axis = axmm(fig, x, top + 19, 29, 12)
        axis.set(xlim=(-.15, 2.15), ylim=(-.48, .70), xticks=[0,1,2], yticks=[-.4,0,.4])
        axis.tick_params(pad=1, labelsize=7)
        if gridrow == 3:
            axis.set_xticklabels(["Amide\ncoupling", "C\u2013N ligand\nlibrary", "C\u2013N metal\nlibrary"], fontsize=5.4)
        else:
            axis.set_xticklabels([])
        if col:
            axis.set_yticklabels([])
        axis.axhline(0, color=LIGHT, linewidth=.5, zorder=0)
        color = DARK if row.plot_label in ("M3", "M8") else ORANGE
        axis.plot([0,1,2], [row.T_amide,row.T_BH1,row.T_BH2], color=color,
                  linewidth=.8, marker="o", markersize=3.1, markeredgewidth=.5,
                  gid="4a_share_" + row.plot_label)
    can.text(1.8, 88, "Condition-preference share, T", fontsize=7, rotation=90,
             ha="center", va="center")

    panel(can, "b", 94, 3, "Owner context: 0.26 vs 0.06")
    can.text(100, 10, "Higher owner share: 17/23", fontsize=7, va="center")
    for y, filled, text in ((14, True, "Owner"), (19, False, "Non-owner")):
        can.plot(158, y, "o", markersize=3.2, markerfacecolor=DARK if filled else "white",
                 markeredgecolor=DARK, markeredgewidth=.6)
        can.text(160, y, text, fontsize=7, va="center", gid="4b_key_" + text)
    axis = axmm(fig, 144, 22, 34, 140)
    axis.set(xlim=(-.52,.88), ylim=(23.5,.5), yticks=[], xticks=[-.4,0,.4,.8])
    axis.spines["left"].set_visible(False)
    axis.axvline(0, color=LIGHT, linewidth=.5, zorder=0)
    for row in molecules.itertuples():
        name = str(row.name_display).replace("$\\mathit{N}$", "N").replace("\n", "")
        lines = textwrap.wrap(name, 28, break_long_words=False, break_on_hyphens=True)
        assert len(lines) <= 2, name
        y = 22 + (row.plot_row-.5) * 140/23
        can.text(95, y, "\n".join(lines), fontsize=6, va="center", linespacing=1.0,
                 gid=f"4b_name_{row.plot_row}")
        color = ORANGE if row.stratum == "amine" else DARK
        axis.plot([row.T_N,row.T_O], [row.plot_row]*2, color=LIGHT, linewidth=.65,
                  gid=f"4b_connector_{row.plot_row}")
        axis.plot(row.T_N,row.plot_row,"o",markersize=3.2,markerfacecolor="white",
                  markeredgecolor=color,markeredgewidth=.6,gid=f"4b_nonowner_{row.plot_row}")
        axis.plot(row.T_O,row.plot_row,"o",markersize=3.2,color=color,
                  markeredgewidth=.5,gid=f"4b_owner_{row.plot_row}")
    can.text(144, 172, "Condition-preference share, T", fontsize=7, ha="center", va="center")
    axis.text(-.50, 1, "Amine", color=ORANGE, fontsize=6, va="center",
              gid="4b_role_amine")
    axis.text(-.36, 20, "Acid", color=DARK, fontsize=6, va="center",
              gid="4b_role_acid")

    panel(can, "c", 1, 180, r"S$_{\mathrm{N}}$Ar: halide 45.6%, N\u2013H 0.3%".replace(r"\u2013", "\u2013"))
    can.text(112, 180, "Base/acid and solvent", fontsize=7, va="center")
    for block, sx, hx, hw, path, centerx in (
        (blocks[0],1,32,56,ROOT/"02_FIGURES/ChemDraw/Fig2_B_halide.svg",13),
        (blocks[1],94,124,54,DISPLAY/"structures/Fig2_c_aliphatic2.svg",105)):
        is_halide = block["key"] == "halide"
        can.text(sx,185,"Fixed halide" if is_halide else "Fixed N\u2013H partner",
                 fontsize=6, color=BLUE if is_halide else ORANGE, va="center")
        structure(can,path,centerx,197,"4c_fixed_"+block["key"])
        heat = axmm(fig,hx,190,hw,16)
        heat.pcolormesh(np.arange(48)-.5,np.arange(len(block["row_ids"])+1)-.5,
                        np.ma.masked_invalid(np.clip(block["z"],-30,30)),
                        cmap=C.HEAT,vmin=-30,vmax=30,shading="flat",
                        edgecolors="none",antialiased=False,gid="4c_fingerprint_"+block["key"])
        heat.set(xlim=(-.5,46.5),ylim=(len(block["row_ids"])-.5,-.5),
                 xticks=[],yticks=range(len(block["row_ids"])))
        heat.set_yticklabels(block["row_ids"],fontsize=5)
        heat.tick_params(length=0,pad=1)
        for spine in heat.spines.values():
            spine.set_visible(False)
        for iy,ix in zip(*np.where(~np.isfinite(block["z"]))):
            from matplotlib.patches import Rectangle
            heat.add_patch(Rectangle((ix-.5,iy-.5),1,1,facecolor="white",edgecolor=LIGHT,
                                     linewidth=.2,hatch="////"))
    can.text(1,211,"Condition preference (pp)",fontsize=7,va="center")
    bar=axmm(fig,75,209,43,1.8)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-30,30),cmap=C.HEAT),cax=bar,orientation="horizontal")
    cb.solids.set_rasterized(False)
    cb.set_ticks([-30,0,30]);cb.ax.tick_params(labelsize=5,length=1.5,pad=1)
    can.text(126,211,"47 conditions; common order",fontsize=6,va="center")
    return fig, dict(figure=4, T_not_Pearson_correlation=True, eight_amine_order=ORDER,
                    reverse_cases=["M8","M3"], molecules=23, owner_higher=17,
                    mean_owner=means[0],mean_nonowner=means[1], SNAr_rows=[len(b["rows"])for b in blocks]), tables, conditions


def role_color(role):
    role=str(role).lower()
    if "halide" in role or "electrophile" in role:
        return BLUE
    if "amine" in role or "n\u2013h" in role or "n-h" in role:
        return ORANGE
    return DARK


def draw5():
    tables=read_tables(5)
    points=tables["A_array_and_pooled"]
    equal=tables["B_equal_depth"]
    curves=tables["C_pooled_curves"]
    thin=tables["C_selected_array_curves"]
    phase2_path=ROOT/"05_DATA/Results/directional/phase2_summary.json"
    nested_path=ROOT/"05_DATA/Results/directional/nested_owner_results.json"
    INPUTS.update([phase2_path,nested_path])
    phase2=json.loads(phase2_path.read_text(encoding="utf-8"))["7.1"]
    nested=json.loads(nested_path.read_text(encoding="utf-8"))
    assert set(curves.targets)=={310}
    assert set(thin.screen)=={"Acylation|1","Suzuki|1"}
    set_a=phase2["Set A"]
    assert (set_a["targets_partner_history_deeper"],set_a["targets_owner_history_deeper"])==(167,121)
    fig,can=figure(202)
    panel(can,"a",1,3,"New pairs: owner > partner")
    axis=axmm(fig,17,15,96,82)
    axis.set(xlim=(-20,29),ylim=(-8,32),xticks=[-20,0,20],yticks=[0,10,20,30])
    axis.axline((0,0),slope=1,color=LIGHT,linewidth=.5,linestyle=(0,(3,3)),zorder=0,
                gid="5a_reference_diagonal")
    axis.axhline(0,color=LIGHT,linewidth=.5,zorder=0,gid="5a_reference_zero_y")
    axis.axvline(0,color=LIGHT,linewidth=.5,zorder=0,gid="5a_reference_zero_x")
    for row in points[points.row_type=="array"].itertuples():
        axis.plot(row.partner_display_pct,row.owner_display_pct,
                  "o" if row.set=="Set A" else "s",markersize=3.8,
                  color=role_color(row.owner_molecular_role),markeredgewidth=.5,
                  gid=f"5a_screen_{row.set}_{row.screen}")
    for row in points[points.row_type=="pooled"].itertuples():
        axis.plot(row.partner_display_pct,row.owner_display_pct,
                  "o" if row.set=="Set A" else "s",color=DARK,markersize=6,
                  markeredgecolor="white",markeredgewidth=.6,zorder=4,gid="5a_pooled_"+row.set)
        offset=(12,4) if row.set=="Set A" else (24,16)
        arrow=dict(arrowstyle="-",color=DARK,lw=.5)
        if row.set=="Set B":
            arrow.update(connectionstyle="angle,angleA=-90,angleB=0,rad=0",relpos=(0,0))
        else:
            arrow.update(connectionstyle="angle,angleA=0,angleB=90,rad=0",relpos=(0,0))
        axis.annotate(row.set,
                      (row.partner_display_pct,row.owner_display_pct),xytext=offset,
                      textcoords="offset points",fontsize=7,va="bottom",linespacing=1.0,
                      arrowprops=arrow,gid="5a_pooled_label_"+row.set)
        if row.set=="Set A":
            axis.annotate("partner record longer:\n167 vs 121 (Set A)",
                          (row.partner_display_pct,row.owner_display_pct),xytext=(8,21),
                          textcoords="offset points",fontsize=7,ha="left",va="bottom",
                          linespacing=1.0,gid="5a_history_depth_SetA")
    axis.set_xlabel("Partner: preference explained (%)",labelpad=3)
    axis.set_ylabel("Owner: preference explained (%)",labelpad=3)
    handles=[]
    for color,label in ((BLUE,"Halide"),(ORANGE,"Amine / N\u2013H"),(DARK,"Other")):
        handle,=axis.plot([],[],"o",markersize=3,color=color,markeredgewidth=.5,label=label)
        handles.append(handle)
    for marker,label in (("o","Set A"),("s","Set B")):
        handle,=axis.plot([],[],marker,markersize=3.8,color=DARK,markeredgewidth=.5,label=label)
        handles.append(handle)
    legend=axis.legend(handles=handles,loc="upper left",bbox_to_anchor=(.025,.70),ncol=5,
                       fontsize=6,handlelength=.8,handletextpad=.5,columnspacing=.8,
                       borderpad=.2,borderaxespad=0,frameon=True,fancybox=False,
                       facecolor="white",edgecolor="white",framealpha=1)
    legend.get_frame().set_linewidth(0)
    legend.set_gid("5a_role_legend")
    for item in legend.get_texts():
        item.set_gid(("5a_set_"if item.get_text().startswith("Set ")else"5a_role_")+item.get_text())

    panel(can,"b",121,3,"Equal depth: owner advantage")
    can.text(126,9,"Two partners per molecule",fontsize=6,va="center")
    for x,filled,label in ((143,False,"Partner"),(160,True,"Owner")):
        can.plot(x,25,"o",markersize=3.8,markerfacecolor=DARK if filled else "white",
                 markeredgecolor=DARK,markeredgewidth=.6,gid="5b_key_handle_"+label)
        can.text(x+2,25,label,fontsize=6,va="center",gid="5b_key_"+label)
    axis=axmm(fig,141,27,30,51)
    axis.patch.set_alpha(0)
    axis.set(xlim=(0,16),ylim=(3,0),xticks=[0,8,16],yticks=[])
    axis.spines["left"].set_visible(False)
    intervals=[]
    for j,row in enumerate(equal.itertuples()):
        y=.4+j
        lab=("Set A","Set B","Within-screen")[j]
        can.text(124,33.8+j*17,lab,fontsize=6,va="center",gid=f"5b_set_{j}")
        axis.plot([row.partner_display_pct,row.owner_display_pct],[y,y],color=LIGHT,linewidth=.7)
        axis.plot(row.partner_display_pct,y,"o",markersize=3.8,markerfacecolor="white",
                  markeredgecolor=DARK,markeredgewidth=.6,gid="5b_partner_"+row.set)
        axis.plot(row.owner_display_pct,y,"o",markersize=3.8,color=DARK,markeredgewidth=.5,gid="5b_owner_"+row.set)
        ci=100*np.asarray(phase2[row.set]["k2"]["ci_whole_array"] if j<2 else nested["fig4b"]["nested"]["D_ci"])
        shown=(10.0,7.3,11.6)[j]
        axis.annotate(f"+{shown:.1f}",(row.owner_display_pct,y),xytext=(5,0),
                      textcoords="offset points",fontsize=6,va="center",gid=f"5b_difference_{j}")
        intervals.append(dict(set=row.set,owner=float(row.owner_display_pct),partner=float(row.partner_display_pct),
                              difference_unrounded=float(row.owner_minus_partner_display_points),
                              approved_display_difference=shown,ci_low=float(ci[0]),ci_high=float(ci[1])))
    can.text(151.5,91,"Preference explained (%); difference (pp)",fontsize=6,ha="center",va="center")

    panel(can,"c",1,111,"Owner histories strengthen with partners")
    can.text(6,117,"Yield profile; amplitude included",fontsize=7,va="center")
    can.text(132,117,"Same 310 held-out pairs",fontsize=7,ha="center",va="center")
    axis=axmm(fig,17,126,151,62)
    axis.set(xlim=(.8,7.7),ylim=(-7,33),xticks=[1,2,3,4,6],yticks=[0,10,20,30])
    axis.axhline(0,color=LIGHT,linewidth=.5,zorder=0)
    axis.text(1.1,-5.9,"dashed: partner",fontsize=6,va="center",gid="5c_partner_style_key")
    for (screen,side),group in thin.groupby(["screen","side"],sort=False):
        group=group.sort_values("partners",kind="stable")
        color=ORANGE if screen=="Acylation|1" else BLUE
        axis.plot(group.partners,group.explained_pct,color=color,linewidth=.65,
                  linestyle="-" if side=="owner" else (0,(3,2)),gid=f"5c_thin_{screen}_{side}")
        last=group.iloc[-1]
        offset=10 if screen=="Acylation|1"and side=="owner"else 0
        axis.annotate("Amide coupling" if screen=="Acylation|1" else "Suzuki",
                      (last.partners,last.explained_pct),xytext=(8,offset),textcoords="offset points",
                      fontsize=7,va="center",gid=f"5c_screen_label_{screen}_{side}")
    for side,group in curves.groupby("side",sort=False):
        group=group.sort_values("partners",kind="stable")
        color=DARK if side=="owner" else LIGHT
        axis.plot(group.partners,group.explained_pct,color=color,linewidth=1.2,
                  linestyle="-" if side=="owner" else (0,(3,2)),marker="o",markersize=2.8,
                  markerfacecolor=color if side=="owner" else "white",markeredgecolor=color,
                  markeredgewidth=.5,zorder=3,gid="5c_pooled_"+side)
        first,last=group.iloc[0],group.iloc[-1]
        axis.annotate(side.capitalize(),
                      (last.partners,last.explained_pct),xytext=(8,0),textcoords="offset points",
                      fontsize=7,va="center",gid="5c_endpoint_"+side)
    axis.set_xlabel("Earlier partners",labelpad=3)
    axis.set_ylabel("Yield profile explained (%)",labelpad=3)
    c.write_csv("Figure5b_equal_depth_intervals.csv",pd.DataFrame(intervals))
    return fig,dict(figure=5,Set_A_pairs=309,Set_B_pairs=535,common_targets=310,
                    SetA_history_depth_scope_only=True,SetA_partner_longer=167,SetA_owner_longer=121,
                    a_b_metric="condition preference with amplitude removed",
                    c_metric="yield profile including response amplitude",equal_depth_intervals=intervals),tables


def inventory(conditions):
    fig,can=figure(90)
    panel(can,"a",1,3,r"S$_{\mathrm{N}}$Ar: 47 base/solvent combinations")
    can.text(6,10,"Measured combinations",fontsize=7,va="center")
    additives=["AcOH","MsOH","PTSA","TFA","none","DIPEA","TEA","2,6-lutidine","Py","DBU",
               "Cs2CO3","K2CO3","KHCO3","KOAc","KOH","KOtBu","NaOMe","NaOtAm"]
    solvents=["DMSO","THF","t-AmOH","NMP","CPME","MeCN"]
    axis=axmm(fig,45,21,100,60)
    axis.set(xlim=(.5,6.5),ylim=(18.5,.5),xticks=range(1,7),yticks=range(1,19))
    axis.set_xticklabels(solvents)
    axis.set_yticklabels(additives)
    axis.xaxis.tick_top();axis.tick_params(length=0,pad=3)
    for spine in axis.spines.values():spine.set_visible(False)
    axis.scatter(conditions["Standardized Name"].map({k:i+1 for i,k in enumerate(solvents)}),
                 conditions.base_or_acid.map({k:i+1 for i,k in enumerate(additives)}),
                 s=12,color=DARK,linewidths=0)
    assert len(conditions)==47
    return fig


def geometry():
    rows=[]
    for path in sorted({Path(r["cdxml"])for r in STRUCTURES}):
        xml=ET.parse(path).getroot()
        nodes={n.get("id"):np.asarray([float(v)for v in n.get("p").split()])
               for n in xml.iter("n")if n.get("p")}
        graph={k:[]for k in nodes}
        lengths=[]
        for bond in xml.iter("b"):
            a,b=bond.get("B"),bond.get("E")
            if a in nodes and b in nodes:
                graph[a].append(b);graph[b].append(a)
                lengths.append(float(np.linalg.norm(nodes[a]-nodes[b])))
        angles=[]
        for center,neighbors in graph.items():
            for a,b in itertools.combinations(neighbors,2):
                v1,v2=nodes[a]-nodes[center],nodes[b]-nodes[center]
                angle=float(np.degrees(np.arccos(np.clip(v1.dot(v2)/np.linalg.norm(v1)/np.linalg.norm(v2),-1,1))))
                angles.append(dict(node=center,neighbors=[a,b],angle_deg=angle,degree=len(neighbors)))
        rows.append(dict(cdxml=str(path),sha256=digest(path),master_BondLength=xml.get("BondLength"),
                         master_LabelSize=xml.get("LabelSize"),median_measured_bond_pt=float(np.median(lengths)),
                         all_bond_lengths_pt=lengths,all_atom_angles=angles,
                         source_geometry_unmodified=True,uniform_print_factor=.7))
        assert abs(float(xml.get("BondLength"))-14.4)<1e-9
        assert abs(float(xml.get("LabelSize"))-10)<1e-9
        assert all(min(abs(a["angle_deg"]-target)for target in(120,180))<.2 for a in angles),path
    return rows


def numeric_checks(t4,t5):
    a=t4["A_eight_amines"]
    assert int((a.T_amide > (a.T_BH1+a.T_BH2)/2).sum())==6
    pooled=t5["A_array_and_pooled"]
    pooled=pooled[pooled.row_type=="pooled"].set_index("set")
    checks=[]
    for name in("Set A","Set B"):
        errors=t5["A_SetA_target_errors"if name=="Set A"else"A_SetB_target_errors"]
        for side,col in(("owner","sse_e"),("partner","sse_n")):
            value=100*(1-errors[col].sum()/errors.sst.sum())
            expected=float(pooled.loc[name,side+"_display_pct"])
            assert np.isclose(value,expected,atol=1e-10,rtol=0),(name,side,value,expected)
            checks.append(dict(panel="5a",set=name,side=side,recomputed=value,retained=expected))
    errors=t5["AB_equal_depth_targets"]
    for row in t5["B_equal_depth"].itertuples():
        target=errors[errors["set"]==row.set]if row.set in("Set A","Set B")else t5["B_within_array_targets"]
        assert len(target)==row.targets
        for side in("owner","partner"):
            col=f"sse_{side}_two_partners"if row.set in("Set A","Set B")else f"sse_{side}"
            value=100*(1-target[col].sum()/target.sst.sum())
            expected=float(getattr(row,side+"_display_pct"))
            assert np.isclose(value,expected,atol=1e-10,rtol=0),(row.set,side,value,expected)
            checks.append(dict(panel="5b",set=row.set,side=side,recomputed=value,retained=expected))
    records=t5["C_retained_depth_records"]
    records=records[records.common_target==True]
    assert len(records)==620
    for row in t5["C_pooled_curves"].itertuples():
        target=records[records.side==row.side]
        assert len(target)==310
        value=100*(1-target[f"sse_{row.partners}"].sum()/target.sst.sum())
        assert np.isclose(value,row.explained_pct,atol=1e-10,rtol=0)
        checks.append(dict(panel="5c",side=row.side,depth=int(row.partners),recomputed=value,
                           retained=float(row.explained_pct)))
    jsave(OUT/"Figure4_5_numeric_QA.json",dict(status="PASS",checks=checks,
          T_definition="variance share, not Pearson correlation",eight_amine_positive=6,
          molecule_context_positive=17,molecule_context_total=23,
          fixed_common_targets_at_all_depths=310))


def annotation_clearance(fig,renderer):
    checks=[]
    for axis in fig.axes:
        pointsets=[]
        for line in axis.lines:
            gid=line.get_gid()or""
            if gid.startswith(("5a_screen_","5a_pooled_","5c_pooled_"))or line.get_marker()=="o":
                if line.get_marker()not in("None","none","",None):
                    xy=axis.transData.transform(np.column_stack([line.get_xdata(),line.get_ydata()]))
                    radius=(float(line.get_markersize())+float(line.get_markeredgewidth()))/2
                    pointsets.extend((pt,radius)for pt in xy)
        for item in axis.texts:
            gid=item.get_gid()or""
            if not gid.startswith(("5a_pooled_label_","5a_history_depth_","5b_value_","5b_difference_",
                                   "5c_endpoint_","5c_screen_label_","5c_partner_style_key")):
                continue
            box=Text.get_window_extent(item,renderer)
            distances=[]
            for xy,radius in pointsets:
                dx=max(box.x0-xy[0],0,xy[0]-box.x1)
                dy=max(box.y0-xy[1],0,xy[1]-box.y1)
                distances.append(np.hypot(dx,dy)*72/fig.dpi-radius)
            minimum=float(min(distances))
            checks.append(dict(gid=gid,text=item.get_text(),nearest_marker_clearance_pt=minimum))
    assert all(c["nearest_marker_clearance_pt"]>=1.99 for c in checks),checks
    return checks


def local_label_clearance(fig, renderer):
    critical=[t for t in fig.findobj(Text)if (t.get_gid()or"").startswith(
        ("4b_role_","5a_history_depth_","5a_role_","5a_set_"))]
    other_text=[t for t in fig.findobj(Text)if t.get_visible()and t.get_text()]
    checks=[]
    for item in critical:
        box=Text.get_window_extent(item,renderer)
        markers=[]
        same_role=[]
        pooled_set_a=[]
        for axis in fig.axes:
            for line in axis.lines:
                gid=line.get_gid()or""
                if not gid.startswith(("4b_owner_","4b_nonowner_","5a_screen_","5a_pooled_")):
                    continue
                xy=axis.transData.transform(np.column_stack([line.get_xdata(),line.get_ydata()]))
                radius=(float(line.get_markersize())+float(line.get_markeredgewidth()))/2
                for point in xy:
                    dx=max(box.x0-point[0],0,point[0]-box.x1)
                    dy=max(box.y0-point[1],0,point[1]-box.y1)
                    gap=float(np.hypot(dx,dy)*72/fig.dpi-radius)
                    markers.append(gap)
                    if gid=="5a_pooled_Set A":
                        pooled_set_a.append(gap)
                    if line.get_markeredgecolor()==item.get_color():
                        same_role.append(gap)
        text_gaps=[]
        for other in other_text:
            if other is item:
                continue
            other_box=Text.get_window_extent(other,renderer)
            dx=max(box.x0-other_box.x1,other_box.x0-box.x1,0)
            dy=max(box.y0-other_box.y1,other_box.y0-box.y1,0)
            text_gaps.append(float(np.hypot(dx,dy)*72/fig.dpi))
        checks.append(dict(gid=item.get_gid(),text=item.get_text(),
                           nearest_data_marker_gap_pt=min(markers),
                           nearest_text_gap_pt=min(text_gaps),
                           nearest_same_role_marker_gap_pt=min(same_role)if same_role else None))
        if item.get_gid()=="5a_history_depth_SetA":
            checks[-1]["pooled_SetA_marker_gap_pt"]=min(pooled_set_a)
            assert min(pooled_set_a)<=21.6,checks[-1]
    assert all(c["nearest_data_marker_gap_pt"]>=1.99 and c["nearest_text_gap_pt"]>=1.99
               for c in checks),checks
    assert all(c["nearest_same_role_marker_gap_pt"]<=21.6
               for c in checks if c["gid"].startswith("4b_role_")),checks
    if any(c["gid"].startswith("5a_role_")for c in checks):
        group_gap=min(c["nearest_data_marker_gap_pt"]for c in checks
                      if c["gid"].startswith("5a_role_"))
        assert group_gap<=21.6,("5a role legend",group_gap)
        for check in checks:
            if check["gid"].startswith("5a_role_"):
                check["legend_group_to_scatter_gap_pt"]=group_gap
    return checks


def annotation_line_clearance(fig,renderer):
    checks=[]
    for axis in fig.axes:
        labels=[t for t in axis.texts if (t.get_gid()or"").startswith(
            ("5a_history_depth_","5a_pooled_label_"))]
        paths=[(line.get_gid(),line.get_transform().transform_path(line.get_path()),line.get_linewidth()/2)
               for line in axis.lines if (line.get_gid()or"").startswith("5a_reference_")]
        paths += [(label.get_gid(),label.arrow_patch.get_transform().transform_path(label.arrow_patch.get_path()),.25)
                  for label in labels if getattr(label,"arrow_patch",None)is not None]
        for label in labels:
            box=Text.get_window_extent(label,renderer)
            corners=np.asarray([[box.x0,box.y0],[box.x0,box.y1],[box.x1,box.y0],[box.x1,box.y1]])
            gaps=[]
            for gid,path,half_width in paths:
                if gid==label.get_gid():
                    continue
                candidates=[]
                for start,end in zip(path.vertices[:-1],path.vertices[1:]):
                    segment=matplotlib.path.Path([start,end])
                    if segment.intersects_bbox(box,filled=False):
                        candidates.append(0.)
                        continue
                    vector=end-start
                    norm=vector.dot(vector)
                    for corner in corners:
                        fraction=np.clip((corner-start).dot(vector)/norm,0,1)if norm else 0
                        candidates.append(float(np.linalg.norm(corner-(start+fraction*vector))))
                    for point in(start,end):
                        dx=max(box.x0-point[0],0,point[0]-box.x1)
                        dy=max(box.y0-point[1],0,point[1]-box.y1)
                        candidates.append(float(np.hypot(dx,dy)))
                gaps.append(dict(line=gid,gap_pt=min(candidates)*72/fig.dpi-half_width))
            checks.append(dict(gid=label.get_gid(),text=label.get_text(),line_gaps=gaps,
                               nearest_independent_line_gap_pt=min(row["gap_pt"]for row in gaps)))
    assert all(row["nearest_independent_line_gap_pt"]>=1.99 for row in checks),checks
    return checks


def save(fig,name,observation,tables=None):
    c.apply_journal(fig)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    clearance=annotation_clearance(fig,renderer)
    local_clearance=local_label_clearance(fig,renderer)
    line_clearance=annotation_line_clearance(fig,renderer)
    return c.save(fig,name,observation,tables,dict(annotation_to_marker_clearance=clearance,
        local_label_clearance=local_clearance,annotation_line_clearance=line_clearance))


def main():
    global OUT
    OUT=c.SOURCES
    with style():
        fig4,obs4,t4,conditions=draw4()
        for sheet,name in (("V6_SNAr_cells","Figure4c_fingerprint_cells.csv"),
                           ("V6_SNAr_row_key","Figure4c_fingerprint_row_key.csv"),
                           ("V6_SNAr_conditions","Figure4c_condition_order.csv")):
            t4[sheet]=pd.read_csv(OUT/name)
        save(fig4,"Figure4",obs4,t4)
    jsave(OUT/"Figure4_structure_geometry.json",geometry())
    return {"panels":{"A":{"eight_amine_positive":6},"B":{"owner_higher":17,"molecules":23},
                      "C":{"halide_rows":4,"NH_rows":8,"conditions":47}}}


if __name__=="__main__":
    c.cli(main,4)

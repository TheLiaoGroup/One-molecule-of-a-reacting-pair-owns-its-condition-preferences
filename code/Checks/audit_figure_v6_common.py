"""Independent v6 tables, live artist readback, provenance and rebuild checks."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"

import numpy as np
import pandas as pd
import pymupdf

import audit_main_v3_values12 as preference
import audit_main_v3_values35 as signed_transfer
import audit_main_v3_values4 as context
import audit_main_v5_values as retained

REFERENCE={2:"screen_shares",3:"preference",4:"molecule_context",5:"transfer"}
PANELS={2:list("ABC"),3:list("abcde"),4:list("ABC"),5:list("ABC")}
HEIGHT={2:215.,3:222.,4:215.,5:202.}
TOL=1e-9
EXTERNAL_SHA="c18d01ebcab0792ed45db6b521aec5fea11c213ca340693c799adb39363399ae"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(actual,expected,label):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    assert a.shape==b.shape and np.allclose(a,b,atol=TOL,rtol=0,equal_nan=True),label


def figure3_display(rb,svg):
    ns={"s":"http://www.w3.org/2000/svg"}
    def lightness(color):
        rgb=[int(color[i:i+2],16)/255 for i in(1,3,5)]
        linear=[x/12.92 if x<=.04045 else((x+.055)/1.055)**2.4 for x in rgb]
        y=sum(x*w for x,w in zip(linear,(.2126729,.7151522,.0721750)))
        return 116*(y**(1/3)if y>(6/29)**3 else y/(3*(6/29)**2)+4/29)-16
    def bounds(group):
        points=[]
        for path in group.findall(".//s:path",ns):
            assert not path.get("transform"),"unexpected transformed colour-scale path"
            values=list(map(float,re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?",path.get("d",""))))
            assert len(values)%2==0,"invalid native mesh coordinates"
            points+=list(zip(values[::2],values[1::2]))
        assert points,"native mesh is empty"
        return [min(x for x,_ in points),min(y for _,y in points),max(x for x,_ in points),max(y for _,y in points)]
    differences=[]
    for path in svg.findall(".//s:path",ns):
        style=dict(part.strip().split(":",1)for part in path.get("style","").split(";")if ":"in part)
        style={key:value.strip().lower()for key,value in style.items()}
        if style.get("fill")=="#cecece"and style.get("stroke")in("#0072b2","#d55e00","#c45a00"):
            differences.append(abs(lightness(style["stroke"])-lightness(style["fill"])))
    assert len(differences)==16 and min(differences)>=30,"D3 selected grid outline/background L* difference is below 30"
    groups={node.get("id"):node for node in svg.findall(".//s:g",ns)}
    anonymous=iter(key for key in groups if re.fullmatch(r"QuadMesh_\d+",key or""))
    scales,heatmaps=[],[]
    for axis in rb["axes"]:
        for mesh in axis.get("meshes",[]):
            gid=mesh.get("gid")or next(anonymous)
            if mesh["clim"]!=[-30,30]:continue
            item=dict(gid=gid,bbox_pt=bounds(groups[gid]))
            (heatmaps if gid.startswith("3b_preference_")and not gid.startswith("3b_preference_scale_")else scales).append(item)
    assert len(heatmaps)==2 and scales,"D11 preference heatmap/scale inventory"
    def distance(a,b):
        return float(np.hypot(max(a[0]-b[2],b[0]-a[2],0),max(a[1]-b[3],b[1]-a[3],0)))
    for heat in heatmaps:
        heat["nearest_scale_gap_pt"]=min(distance(heat["bbox_pt"],scale["bbox_pt"])for scale in scales)
        assert heat["nearest_scale_gap_pt"]<=21.6,"D11 preference scale is farther than 0.3 inch from its heatmap"
    return dict(selected_grid_outlines=16,minimum_Lstar_difference=min(differences),
                preference_scale_distances=heatmaps,scale_scope="Preference, -30 to +30 pp; matrix r uses its own local range label")


def rewrite(value):
    if not isinstance(value,str):return value
    for number,name in {1:"overview",2:"preference",3:"screen_shares",4:"molecule_context",5:"transfer"}.items():
        value=value.replace(f"04_SOURCE_DATA/Figure{number}/Figure{number}_SourceData.xlsx",
                            f"05_DATA/Analysis/figure_display/v5_reference_tables/{name}.xlsx")
    return value.replace("04_SOURCE_DATA/Figure2/figure2_payload.json",
                         "05_DATA/Analysis/figure_display/v5_reference_tables/preference_payload.json")


def items(rb,kind):
    rows=[row for axis in rb["axes"] for row in axis.get(kind,[])if row.get("gid")]
    assert len({row["gid"]for row in rows})==len(rows),"duplicate quantitative artist identity"
    return {row["gid"]:row for row in rows}


def retained_cells(number,root,tables):
    folder=root/"05_DATA/Analysis/figure_display/v5_reference_tables"
    manifest=load(folder/"MANIFEST.json")
    path=folder/f"{REFERENCE[number]}.xlsx"
    assert digest(path) in json.dumps(manifest),"frozen reference workbook hash"
    reference=retained.read_stored_workbook(path)
    count=0
    for sheet,expected in reference.items():
        assert sheet in tables,"missing retained sheet: "+sheet
        actual=tables[sheet]
        assert actual.shape==expected.shape and actual.columns.tolist()==expected.columns.tolist(),"retained shape: "+sheet
        for column in expected:expected[column]=expected[column].map(rewrite)
        assert all(retained.retained_cell_equal(a,b)for a,b in zip(actual.to_numpy(object).flat,expected.to_numpy(object).flat)),"retained cell: "+sheet
        count+=actual.size
    return dict(sheets=len(reference),cells=count,exact=True)


@contextmanager
def selected_paths(number,artifact):
    original=signed_transfer._path
    def adapted(root,*names):
        if number==2 and any(name.endswith("Figure3/figure3_selection.json")for name in names):
            return artifact/"figure2_selection.json"
        if number==5 and any(name.endswith("Figure5/figure5_selection.json")for name in names):
            return artifact/"figure5_selection.json"
        return original(root,*names)
    signed_transfer._path=adapted
    try:yield
    finally:signed_transfer._path=original


def independent(number,root,tables,artifact):
    if number in (2,5):
        with selected_paths(number,artifact):
            return signed_transfer.audit_tables(3 if number==2 else 5,root,tables)
    if number==4:
        with retained.directional_public_reader(3):
            return context.audit_tables(4,root,tables)
    evidence=preference.Evidence()
    ref=preference._preference_reference(root,evidence)
    payload=load(root/"05_DATA/Analysis/figure_display/v6_figure3/aminations_payload.json")
    assert payload["condition_order"]==ref["order"],"Figure3 shared condition order"
    for role in ("amine","halide"):
        actual,expected=payload[role+"_rows"],ref["partners"][role]
        assert [row["smiles"]for row in actual]==[row["smiles"]for row in expected],"Figure3 molecule order"
        close([row["pattern_points"]for row in actual],[row["pattern_points"]for row in expected],"Figure3 independently recomputed 95-column preferences")
    agreement=pd.read_csv(root/"05_DATA/Analysis/figure_display/v6_figure3/all_molecule_agreement.csv")
    agreement=agreement[agreement.row_type.eq("molecule")]
    observed=tables["C_molecule_agreement"]
    assert len(observed)==len(agreement),"Figure3 all eligible molecule rows"
    for name in ("cn","amide"):
        draws=pd.read_csv(root/f"05_DATA/Analysis/figure_display/v6_figure3/class_null/{name}_draws.csv")
        close(tables["V6_class_null_CN"if name=="cn"else"V6_class_null_amide"].explained,draws.explained,"Figure3 class-null draw inventory")
    return evidence.result(condition_columns=95,heatmap_rows=17)


def artist_values(number,root,tables,rb,artifact):
    lines,collections,patches,meshes=(items(rb,kind)for kind in("lines","collections","patches","meshes"))
    if number==2:
        arrays=tables["Fig3A_arrays"]
        layout=rb["metadata"]["layout"]
        assert len(layout)==len(arrays)==15,"Figure2 screen inventory"
        for i,(row,placement)in enumerate(zip(arrays.itertuples(),layout),1):
            for side,offset in (("owner",-1.),("partner",1.)):
                patch=patches[f"a_{side}_{i}"]
                close([patch["width"]],[100*getattr(row,side+"_share")],"Figure2 signed bar value")
                close([patch["xy"][1]+patch["height"]/2],[placement["center_y_mm"]+offset],"Figure2 selected row alignment")
            assert placement["screen"]==row.screen,"Figure2 selected screen"
        for field in("share","owner_fraction"):
            values=100*arrays[field].astype(float).to_numpy()
            visible=values<=150
            close(np.asarray(collections["b_points_"+field]["offsets"])[:,1],values[visible],"Figure2 distribution values")
            close(lines["b_median_"+field]["y"],[np.median(values)]*2,"Figure2 median")
            if (~visible).any():
                close(np.asarray(collections["b_points_"+field+"_overflow"]["offsets"])[:,1],np.repeat(150,(~visible).sum()),"Figure2 overflow marker")
            rows=tables["V6_summary_points"]
            rows=rows[rows.quantity.eq(field)]
            assert rows.screen.tolist()==arrays.screen.tolist(),"Figure2 summary source identities"
            close(rows.value_pct,values,"Figure2 summary source values")
        source=next((root/row["file"]for group in load(artifact/"Figure2_SourceData_provenance.json")["inputs"].values()for row in group.values()if "external_replication/"in row["file"]),None)
        assert source is not None and source.is_file(),"Figure2 external input record"
        assert digest(source)==EXTERNAL_SHA,"Figure2 frozen external identity/value payload"
        data=load(source)
        for dataset,reference in (("CH_arylation","BH_ligand"),("Suzuki","Suzuki")):
            for role in("halide","partner"):
                expected=[r["consistency"]for r in data["molecules"][dataset]if r["status"]=="eligible"and r["role"]==role]
                close(np.asarray(collections[f"c_points_{dataset}_{role}"]["offsets"])[:,1],expected,"Figure2 eligible external molecule points")
                close(lines[f"c_median_{dataset}_{role}"]["y"],[np.median(expected)]*2,"Figure2 external median")
                close(lines[f"c_reference_{dataset}_{role}"]["y"],[data["benchmarks"][reference][role]]*2,"Figure2 P2 reference")
    elif number==3:
        payload=load(root/"05_DATA/Analysis/figure_display/v6_figure3/aminations_payload.json")
        assert rb["metadata"]["frozen_order"]==payload["condition_order"],"Figure3 frozen display order"
        for role in("amine","halide"):
            data=np.asarray([row["pattern_points"]for row in payload[role+"_rows"]],float)
            close(meshes["3b_preference_"+role]["array"],data,"Figure3 complete preference mesh")
            expected=np.full((len(data),len(data)),np.nan)
            for i in range(len(data)):
                for j in range(i):
                    good=np.isfinite(data[i])&np.isfinite(data[j])
                    expected[i,j]=expected[j,i]=np.corrcoef(data[i,good],data[j,good])[0,1]
            close(meshes["3b_correlations_"+role]["array"],expected,"Figure3 complete pairwise matrix")
            r=[row["r"]for row in payload["correlations"]if row["role"]==role]
            close(np.asarray(collections["3b_pairwise_"+role]["offsets"])[:,0],r,"Figure3 pairwise points")
            close(lines["3b_mean_"+role]["x"],[np.mean(r)]*2,"Figure3 mean agreement")
        data=pd.read_csv(root/"05_DATA/Analysis/figure_display/v6_figure3/all_molecule_agreement.csv")
        data=data[data.row_type.eq("molecule")]
        for screen,role in(("Buchwald-Hartwig|1",2),("Buchwald-Hartwig|1",1),("Acylation|1",1),("Acylation|1",2)):
            values=data[data.screen.eq(screen)&data.role.eq(role)].agreement.to_numpy(float)
            close(np.asarray(collections[f"3c_points_{screen}_{role}"]["offsets"])[:,0],values,"Figure3 molecule agreements")
            close(lines[f"3c_median_{screen}_{role}"]["x"],[np.median(values)]*2,"Figure3 group median")
        summaries=load(root/"05_DATA/Analysis/figure_display/v6_figure3/class_null/summary.json")
        observed_summary=tables["V6_class_null_summary"]
        assert observed_summary.screen.tolist()==[r["screen"]for r in summaries],"Figure3 class summary identities"
        for field in("seed","draws","explained","null_median","null_p95","p","pairs"):
            close(observed_summary[field],[r[field]for r in summaries],"Figure3 class summary values")
        for summary,name in zip(summaries,("cn","amide")):
            draws=pd.read_csv(root/f"05_DATA/Analysis/figure_display/v6_figure3/class_null/{name}_draws.csv").explained.to_numpy()
            assert len(draws)==summary["draws"]==4999,"Figure3 shuffle count"
            close([summary["null_median"]],[np.median(draws)],"Figure3 shuffled median")
            close(lines["3d_null_median_"+name]["x"],[np.median(draws)]*2,"Figure3 plotted shuffled median")
            close(np.asarray(collections["3d_observed_"+name]["offsets"])[:,0],[summary["explained"]],"Figure3 observed class score")
    elif number==4:
        for row in tables["A_eight_amines"].itertuples():
            close(lines["4a_share_"+row.plot_label]["y"],[row.T_amide,row.T_BH1,row.T_BH2],"Figure4 triple-context T")
        for row in tables["B_23_molecules"].itertuples():
            for label,field in(("owner","T_O"),("nonowner","T_N")):
                close(lines[f"4b_{label}_{row.plot_row}"]["x"],[getattr(row,field)],"Figure4 molecule context point")
        selection=load(artifact/"Figure4c_selection.json")
        assert selection["fixed_NH"]=="CC1CCCNC1"and selection["fixed_halide"]=="Cc1cc(F)ccc1C#N","Figure4 frozen focal molecules"
        evidence=preference.Evidence()
        patterns=preference._patterns(preference._raw(root,evidence),"SNAr|1")
        cells=tables["V6_SNAr_cells"]
        row_key=tables["V6_SNAr_row_key"]
        conditions=tables["V6_SNAr_conditions"]
        assert len(conditions)==conditions.condition_id.nunique()==47,"Figure4 complete condition inventory"
        assert conditions.condition_id.tolist()==selection["condition_order"],"Figure4 frozen condition ordering"
        assert len(row_key)==len(row_key[["block","row_id"]].drop_duplicates()),"Figure4 unique fingerprint row identities"
        assert len(cells)==12*47 and len(row_key)==12,"Figure4 fingerprint row/column coverage"
        assert len(cells)==len(cells[["block","row_id","display_column"]].drop_duplicates()),"Figure4 unique fingerprint cells"
        for (_, _),group in cells.groupby(["block","row_id"],sort=False):
            assert group.condition_id.tolist()==conditions.condition_id.tolist() and group.display_column.tolist()==list(range(1,48)),"Figure4 shared cell-to-condition ordering"
        candidates=sorted([(int((patterns["a"]==s).sum()),str(s))for s in set(patterns["a"])],key=lambda r:(-r[0],r[1]))
        assert candidates[0]==(8,selection["fixed_NH"]),"Figure4 largest productive-partner SNAr rule"
        for block,fixed,side in(("halide",selection["fixed_halide"],"b"),("NH",selection["fixed_NH"],"a")):
            rows=cells[cells.block.eq(block)]
            keys=row_key[row_key.block.eq(block)]
            assert set(keys.row_id)==set(rows.row_id)and keys.fixed_smiles.eq(fixed).all(),"Figure4 row-key focal molecule selection"
            for key in keys.itertuples():
                assert rows[rows.row_id.eq(key.row_id)].source_subpair.eq(key.source_subpair).all(),"Figure4 row-key source-pair selection"
            matrix=rows.pivot(index="row_id",columns="display_column",values="clipped_pp")
            close(meshes["4c_fingerprint_"+block]["array"],matrix.to_numpy(),"Figure4 fingerprint mesh")
            for row in rows.itertuples():
                i=list(patterns["pairs"]).index(row.source_subpair)
                j=patterns["conditions"].index(row.condition_id)
                assert patterns[side][i]==fixed,"Figure4 fingerprint focal identity"
                close([row.preference_pp],[patterns["z"][i,j]],"Figure4 independently recomputed SNAr preference")
    else:
        for row in tables["A_array_and_pooled"].itertuples():
            gid="5a_pooled_"+row.set if row.row_type=="pooled"else f"5a_screen_{row.set}_{row.screen}"
            close(lines[gid]["x"],[row.partner_display_pct],"Figure5 per-array partner value")
            close(lines[gid]["y"],[row.owner_display_pct],"Figure5 per-array owner value")
        for row in tables["B_equal_depth"].itertuples():
            for side in("owner","partner"):
                close(lines[f"5b_{side}_{row.set}"]["x"],[getattr(row,side+"_display_pct")],"Figure5 equal-depth value")
        for side,group in tables["C_pooled_curves"].groupby("side",sort=False):
            group=group.sort_values("partners",kind="stable")
            close(lines["5c_pooled_"+side]["x"],group.partners,"Figure5 retained depths")
            close(lines["5c_pooled_"+side]["y"],group.explained_pct,"Figure5 common-target pooled curve")
        for (screen,side),group in tables["C_selected_array_curves"].groupby(["screen","side"],sort=False):
            group=group.sort_values("partners",kind="stable")
            close(lines[f"5c_thin_{screen}_{side}"]["y"],group.explained_pct,"Figure5 selected-array curves")
        errors=tables["A_SetA_target_errors"].copy()
        errors["gain"]=errors.sse_n-errors.sse_e
        array_groups=errors.groupby("screen")[["gain","sst"]].sum()
        reference=load(root/"05_DATA/Results/reserve_validation/reserve_validation.json")["e2"]
        for filename in("SetA_whole_array_bootstrap_draws.csv","SetA_owner_block_bootstrap_draws.csv"):
            draws=pd.read_csv(artifact/filename)
            assert len(draws)==2000 and draws.draw.tolist()==list(range(1,2001)),"Figure5 full SetA bootstrap draw coverage"
            if "whole_array"in filename:
                names=draws.drop(columns=["draw","D"]).to_numpy()
                recomputed=array_groups.gain.loc[names.ravel()].to_numpy().reshape(names.shape).sum(1)/array_groups.sst.loc[names.ravel()].to_numpy().reshape(names.shape).sum(1)
                interval=reference["ci_whole_array"]
            else:
                numerator,denominator=np.zeros(len(draws)),np.zeros(len(draws))
                for screen,group in errors.groupby("screen"):
                    blocks=group.groupby("electrophile")[["gain","sst"]].sum()
                    for i,value in enumerate(draws[screen]):
                        owners=value.split(";")
                        assert len(owners)==len(blocks),"Figure5 owner bootstrap block depth"
                        selected=blocks.loc[owners]
                        numerator[i]+=selected.gain.sum()
                        denominator[i]+=selected.sst.sum()
                recomputed=numerator/denominator
                interval=reference["ci_electrophile_cluster"]
            close(draws.D,recomputed,"Figure5 independently recomputed every SetA bootstrap draw")
            close(np.percentile(recomputed,[2.5,97.5]),interval,"Figure5 retained SetA bootstrap interval")
    return dict(quantitative_lines=len(lines),quantitative_collections=len(collections),quantitative_meshes=len(meshes),quantitative_patches=len(patches))


def visual(number,rb,journal):
    assert rb["figure"]==number and rb["journal"]==journal,"figure/journal identity"
    close([rb["width_mm"],rb["height_mm"]],[180,HEIGHT[number]],"figure dimensions")
    labels=[row for row in rb["labels"]if (row.get("gid")or"").startswith("panel_")and len(row["gid"])==7]
    expected=[s.lower()for s in PANELS[number]]if journal=="natcatal"else PANELS[number]
    assert [row["text"]for row in labels]==expected,"journal panel inventory/case"
    assert all(row["fontsize"]==(8 if journal=="natcatal"else 9)and row["weight"]in("bold",700)for row in labels),"journal panel typography"
    assert all(5-1e-6<=row["fontsize"]<=9+1e-6 for row in rb["texts"]),"minimum font size"
    if number==2:
        assert rb["clearance"]["minimum_text_to_point_clearance_pt"]>=2,"Figure2 point clearance"
        assert rb["clearance"]["minimum_text_to_data_line_clearance_pt"]>=2,"Figure2 line clearance"
    if number in(4,5):
        for row in rb["annotation_to_marker_clearance"]:
            assert row["nearest_marker_clearance_pt"]>=1.99,"annotation marker clearance"
        for row in rb["local_label_clearance"]:
            assert row["nearest_data_marker_gap_pt"]>=1.99 and row["nearest_text_gap_pt"]>=1.99,"local label clearance"
        for row in rb["annotation_line_clearance"]:
            assert row["nearest_independent_line_gap_pt"]>=1.99,"independent line clearance"
    return dict(panels=expected,minimum_font_pt=min(row["fontsize"]for row in rb["texts"]))


def exported(number,pdf,svg):
    document=pymupdf.open(pdf)
    assert len(document)==1 and not document[0].get_images(full=True),"vector-only single-page PDF"
    spans=[s for b in document[0].get_text("dict")["blocks"]for l in b.get("lines",[])for s in l.get("spans",[])if s["text"].strip()]
    assert all("Arial"in s["font"]and s["size"]>=4.999 and document[0].rect.contains(pymupdf.Rect(s["bbox"]))for s in spans),"PDF font/on-page verification"
    assert not any(node.tag.endswith("}image")for node in ET.parse(svg).iter()),"SVG contains raster image"
    document.close()
    return dict(vector_pdf=True,editable_svg=True,pdf_text_spans=len(spans))


def provenance(number,root,artifact,record,output_path):
    assert record["figure"]==f"Figure {number}"and record["generated_by"]==f"06_CODE/Figures/figure{number}.py","provenance figure owner"
    inputs=[row for group in record["inputs"].values()for row in group.values()]
    assert inputs and len({r["file"]for r in inputs})==len(inputs),"input coverage"
    assert any(r["file"]==f"06_CODE/Figures/figure{number}.py"for r in inputs),"generator provenance"
    if number==2:assert any("external_replication/"in r["file"]for r in inputs),"external provenance record missing"
    if number==3:assert any("v6_figure3/aminations_payload.json"in r["file"]for r in inputs),"amination display input missing"
    if number==4:assert any("v6_figure4/SNAr_display_rule.md"in r["file"]for r in inputs),"SNAr display rule input missing"
    if number==5:assert any("Fig4a_folds.csv"in r["file"]for r in inputs),"transfer fold provenance missing"
    for row in inputs:assert digest(root/row["file"])==row["sha256"],"input provenance hash: "+row["file"]
    outputs=list(record["outputs"].values())
    assert outputs and len({r["file"].casefold()for r in outputs})==len(outputs),"unique output ownership"
    for row in outputs:assert digest(output_path(row["file"]))==row["sha256"],"output provenance hash: "+row["file"]
    assert not any("external_points"in r["file"]for r in outputs),"external raw derivative exported"
    return dict(inputs=len(inputs),outputs=len(outputs),hashes_verified=True)


def fault_controls(number,root,tables,rb,artifact,prov,journal,output_path,svg_path):
    faults=[]
    def reject(label,fn):
        try:fn()
        except (AssertionError,KeyError)as error:
            faults.append(dict(fault=label,rejected=True,reason=str(error)))
            return
        raise AssertionError("fault accepted: "+label)
    bad={k:v.copy(deep=True)for k,v in tables.items()}
    sheet={2:"Fig3A_arrays",3:"A_correlations",4:"A_eight_amines",5:"A_array_and_pooled"}[number]
    column=next(c for c in bad[sheet]if any(isinstance(v,(int,float,np.number))and not isinstance(v,(bool,np.bool_))for v in bad[sheet][c].dropna()))
    index=next(i for i,v in bad[sheet][column].items()if isinstance(v,(int,float,np.number))and not isinstance(v,(bool,np.bool_)))
    bad[sheet].loc[index,column]=float(bad[sheet].loc[index,column])+1
    reject("numeric source corruption",lambda:retained_cells(number,root,bad))
    bad={k:v.copy(deep=True)for k,v in tables.items()}
    bad[sheet]=bad[sheet].iloc[:-1]
    reject("missing source row",lambda:retained_cells(number,root,bad))
    if number in(4,5):
        bad={k:v.copy(deep=True)for k,v in tables.items()}
        bad["Glyph_actual_barcode"].loc[0,"source_file"]="04_SOURCE_DATA/Figure2/figure2_payload.json"
        reject("historical source pointer",lambda:retained_cells(number,root,bad))
    bad=copy.deepcopy(rb)
    next(row for row in bad["labels"]if (row.get("gid")or"").startswith("panel_"))["text"]="Z"
    reject("wrong panel",lambda:visual(number,bad,journal))
    bad=copy.deepcopy(rb)
    next(row for row in bad["labels"]if (row.get("gid")or"").startswith("panel_"))["fontsize"]=6
    reject("wrong panel font",lambda:visual(number,bad,journal))
    bad=copy.deepcopy(rb)
    gid,field={2:("b_median_share","y"),3:("3b_mean_amine","x"),4:("4a_share_M4","y"),5:("5a_pooled_Set A","x")}[number]
    target=next(row for axis in bad["axes"]for row in axis["lines"]if row.get("gid")==gid)
    target[field][0]+=1
    reject("actual artist corruption",lambda:artist_values(number,root,tables,bad,artifact))
    if number in(2,4,5):
        bad={k:v.copy(deep=True)for k,v in tables.items()}
        if number==2:
            bad["Rule2_selected"].loc[0,"smiles"]="C"
            reject("frozen selected molecule corruption",lambda:independent(number,root,bad,artifact))
        elif number==4:
            bad["V6_SNAr_row_key"].loc[0,"fixed_smiles"]="C"
            reject("SNAr selected molecule corruption",lambda:artist_values(number,root,bad,rb,artifact))
        else:
            bad["C_selection_counts"]["thin_curve"]=True
            reject("selected low-count array corruption",lambda:independent(number,root,bad,artifact))
    else:
        bad=copy.deepcopy(rb)
        bad["metadata"]["frozen_order"]=list(reversed(bad["metadata"]["frozen_order"]))
        reject("frozen condition-order corruption",lambda:artist_values(number,root,tables,bad,artifact))
        svg=ET.parse(svg_path).getroot()
        bad=copy.deepcopy(svg)
        for node in bad.iter():
            if node.get("style"):node.set("style",node.get("style").replace("#c45a00","#d55e00"))
        reject("old low-contrast orange outline",lambda:figure3_display(rb,bad))
        bad=copy.deepcopy(svg)
        ns={"s":"http://www.w3.org/2000/svg"}
        for group in bad.findall(".//s:g",ns):
            if not(group.get("id")or"").startswith("3b_preference_scale_"):continue
            for path in group.findall(".//s:path",ns):
                values=list(map(float,re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?",path.get("d",""))))
                points=list(zip(values[::2],values[1::2]))
                path.set("d","M "+" L ".join(f"{x} {y+55*72/25.4}"for x,y in points)+" z")
        reject("old distant preference-scale position",lambda:figure3_display(rb,bad))
    bad=copy.deepcopy(prov)
    next(row for group in bad["inputs"].values()for row in group.values())["sha256"]="0"*64
    reject("input provenance corruption",lambda:provenance(number,root,artifact,bad,output_path))
    if number==2:
        bad=copy.deepcopy(prov)
        for group in bad["inputs"].values():
            for key in list(group):
                if "external_replication/"in group[key]["file"]:del group[key]
        reject("external record missing",lambda:provenance(number,root,artifact,bad,output_path))
    return faults


def run(number):
    if number==3:
        from audit_figure3_v7 import run as run_v7
        return run_v7()
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument("--qc-dir",type=Path)
    parser.add_argument("--artifact-dir",type=Path)
    parser.add_argument("--journal",choices=("science","natcatal"),default="science")
    parser.add_argument("--values-only",action="store_true")
    args=parser.parse_args()
    root=args.root.resolve()
    qc=(args.qc_dir or root/"08_REVIEW_AND_QC").resolve()
    qc.mkdir(parents=True,exist_ok=True)
    base=root/"02_FIGURES/Journal_Formats/natcatal"if args.journal=="natcatal"else root
    artifact=args.artifact_dir.resolve()if args.artifact_dir else (base/f"Figure{number}"if args.journal=="natcatal"else root/f"04_SOURCE_DATA/Figure{number}")
    def output_path(relative):
        if args.artifact_dir:return artifact/Path(relative).name
        if args.journal=="natcatal":
            return base/("Supplementary_Figures"if "Supplementary_Figures"in relative else "Main"if "/Main/"in relative else f"Figure{number}")/Path(relative).name
        return root/relative
    rb=load(artifact/f"Figure{number}_readback.json")
    tables=retained.read_stored_workbook(artifact/f"Figure{number}_SourceData.xlsx")
    prov=load(artifact/f"Figure{number}_SourceData_provenance.json")
    checks={"retained_cells":retained_cells(number,root,tables),
            "independent_values":independent(number,root,tables,artifact),
            "artists":artist_values(number,root,tables,rb,artifact),
            "journal":visual(number,rb,args.journal),
            "export":exported(number,output_path(f"02_FIGURES/Main/Figure{number}.pdf"),output_path(f"02_FIGURES/Main/Figure{number}.svg")),
            "provenance":provenance(number,root,artifact,prov,output_path)}
    svg_path=output_path(f"02_FIGURES/Main/Figure{number}.svg")
    if number==3:checks["D3_D11_display"]=figure3_display(rb,ET.parse(svg_path).getroot())
    faults=fault_controls(number,root,tables,rb,artifact,prov,args.journal,output_path,svg_path)
    if not args.values_only:
        with tempfile.TemporaryDirectory(prefix=f"v6_figure{number}_",dir=qc)as folder:
            command=[sys.executable,str(root/f"06_CODE/Figures/figure{number}.py"),"--out",folder,"--journal",args.journal]
            completed=subprocess.run(command,cwd=root,capture_output=True,text=True,encoding="utf-8",errors="replace")
            assert completed.returncode==0,completed.stdout+completed.stderr
            fresh=load(Path(folder)/f"Figure{number}_SourceData_provenance.json")
            assert fresh==prov,"isolated provenance differs"
            for row in prov["outputs"].values():assert digest(Path(folder)/Path(row["file"]).name)==row["sha256"],"isolated artifact differs: "+row["file"]
            checks["fresh_rebuild"]=dict(result="PASS",outputs=len(prov["outputs"]),deterministic=True)
    result=dict(result="PASS",figure=number,journal=args.journal,checks=checks,fault_injections=faults)
    destination=qc/f"Figure{number}_v6_audit.json"
    destination.write_text(json.dumps(result,indent=2,ensure_ascii=True)+"\n",encoding="utf-8")
    print(json.dumps(dict(result="PASS",figure=number,journal=args.journal,faults_rejected=len(faults),report=str(destination)),indent=2))


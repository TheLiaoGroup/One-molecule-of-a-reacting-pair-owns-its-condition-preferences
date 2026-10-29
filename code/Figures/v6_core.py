"""Portable exports for the approved five-figure v6 presentation."""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "06_CODE/Figures"))
import structure_art
from figure3_selection import load_art

BLUE, ORANGE, DARK, LIGHT = "#0072B2", "#D55E00", "#4D4D4D", "#B3B3B3"
HEAT = LinearSegmentedColormap.from_list("v6_preference", ["#5A5A5A", "#FFFFFF", "#D9B748"])
PT_MM = 25.4 / 72
WIDTH = 180
JOURNAL = "science"
OUTPUT = ROOT / "02_FIGURES/Main"
SOURCES = ROOT / "04_SOURCE_DATA"
SUPPLEMENTARY = ROOT / "02_FIGURES/Supplementary_Figures"
NUMBER = None
OVERRIDE = None
EXTERNAL_RESULT = None
OUTPUT_RECORDS = {}


@contextmanager
def style():
    with plt.rc_context({
        "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
        "axes.titlesize": 7, "axes.linewidth": .5,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": .5, "ytick.major.width": .5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "lines.linewidth": .65, "pdf.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "p2-v6-draft",
        "mathtext.fontset": "custom", "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.grid": False,
    }):
        yield


def axes(fig, x, y, w, h):
    height = fig.get_figheight() * 25.4
    return fig.add_axes([x / WIDTH, 1 - (y + h) / height, w / WIDTH, h / height])


def new(height):
    fig = plt.figure(figsize=(WIDTH / 25.4, height / 25.4), dpi=180)
    canvas = axes(fig, 0, 0, WIDTH, height)
    canvas.set(xlim=(0, WIDTH), ylim=(height, 0))
    canvas.axis("off")
    return fig, canvas


def panel(canvas, letter, x, y, tag=None):
    canvas.text(x, y, letter, size=8, weight="bold", va="center", gid=f"panel_{letter}")
    if tag:
        canvas.text(x + 5, y, tag, size=7, va="center", gid=f"tag_{letter}")


def structure(canvas, path, x, y, gid):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    size = structure_art.draw(canvas, load_art(path), x, y, 7, gid)
    for patch in canvas.patches:
        if (patch.get_gid() or "").startswith(gid + "|bond"):
            patch.set_edgecolor("black")
            patch.set_linewidth(.08)
    return size


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tables(number):
    import main_figure_common as retained
    previous = {2: 3, 3: 2, 4: 4, 5: 5}[number]
    result = retained.read_tables(previous)
    for frame in result.values():
        for column in frame:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(retained_source)
    return result


def retained_source(value):
    if not isinstance(value, str):
        return value
    names = {1: "overview", 2: "preference", 3: "screen_shares", 4: "molecule_context", 5: "transfer"}
    for number, name in names.items():
        value = value.replace(f"04_SOURCE_DATA/Figure{number}/Figure{number}_SourceData.xlsx",
                              f"05_DATA/Analysis/figure_display/v5_reference_tables/{name}.xlsx")
    return value.replace("04_SOURCE_DATA/Figure2/figure2_payload.json",
                         "05_DATA/Analysis/figure_display/v5_reference_tables/preference_payload.json")


def apply_journal(fig):
    for item in fig.findobj(Text):
        gid=item.get_gid()or""
        if re.fullmatch(r"panel_[a-zA-Z]",gid):
            letter=gid[-1]
            item.set_text(letter.lower()if JOURNAL=="natcatal"else letter.upper())
            item.set_fontsize(8 if JOURNAL=="natcatal"else 9)
            item.set_fontweight("bold")
            item.set_fontstyle("normal")
        elif JOURNAL=="science":
            item.set_text(re.sub(r"\b(Fig\. \d+)([a-z])\b",lambda m:m[1]+m[2].upper(),item.get_text()))


def write_json(name, value):
    from figure_export import clean
    def portable(item):
        if isinstance(item,dict):return {portable(str(k)):portable(v)for k,v in item.items()}
        if isinstance(item,list):return [portable(v)for v in item]
        if isinstance(item,str):
            return item.replace(str(ROOT)+os.sep,"").replace(str(ROOT).replace("\\","/")+"/","").replace("\\","/")
        return item
    SOURCES.mkdir(parents=True,exist_ok=True)
    path=SOURCES/name
    path.write_text(json.dumps(portable(clean(value)),ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    key="readback"if name==f"Figure{NUMBER}_readback.json"else name
    OUTPUT_RECORDS[key]=(ROOT/f"04_SOURCE_DATA/Figure{NUMBER}"/name,path)
    return path


def write_csv(name, frame):
    SOURCES.mkdir(parents=True,exist_ok=True)
    path=SOURCES/name
    frame.to_csv(path,index=False,lineterminator="\n")
    OUTPUT_RECORDS[name]=(ROOT/f"04_SOURCE_DATA/Figure{NUMBER}"/name,path)
    return path


def save(fig, name, metadata=None, source_tables=None, extra_readback=None):
    import pymupdf as fitz
    from figure_export import capture_readback, clean
    from figure1_data import write_xlsx
    apply_journal(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = []
    for text in fig.findobj(Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            labels.append(dict(text=text.get_text(), gid=text.get_gid(),
                               fontsize=text.get_fontsize(),
                               weight=text.get_fontweight(),
                               bbox_pt=[v * 72 / fig.dpi for v in box.extents]))
    supplemental=name.startswith("FigS")
    output=SUPPLEMENTARY if supplemental else OUTPUT
    output.mkdir(parents=True,exist_ok=True)
    for ext in("svg","pdf","png"):
        kw={"metadata":{"Date":None,"Creator":"P2 v6 formal"}}if ext=="svg"else{}
        if ext=="pdf":kw={"metadata":{"CreationDate":None,"ModDate":None,"Creator":"P2 v6 formal"}}
        if ext=="png":kw={"dpi":600,"metadata":{"Software":"P2 v6 formal"}}
        fig.savefig(output/f"{name}.{ext}",**kw)
        if supplemental:
            OUTPUT_RECORDS[f"{name}_{ext}"]=(ROOT/"02_FIGURES/Supplementary_Figures"/f"{name}.{ext}",output/f"{name}.{ext}")
        else:
            OUTPUT_RECORDS[ext]=(ROOT/"02_FIGURES/Main"/f"{name}.{ext}",output/f"{name}.{ext}")
    pdf = fitz.open(output / f"{name}.pdf")
    spans = [span for page in pdf for block in page.get_text("dict")["blocks"]
             for line in block.get("lines", []) for span in line.get("spans", [])]
    font_sizes = sorted(set(round(span["size"], 4) for span in spans if span["text"].strip()))
    assert min(font_sizes) >= 5 - 1e-3, font_sizes
    assert max(font_sizes) <= (8 if JOURNAL=="natcatal"else 9) + 1e-3,font_sizes
    assert len(pdf)==1 and not pdf[0].get_images(full=True),name
    assert all("Arial"in span["font"]for span in spans if span["text"].strip()),name
    assert all(pdf[0].rect.contains(fitz.Rect(span["bbox"]))for span in spans if span["text"].strip()),name
    titles={t.get_gid():t.get_text()for t in fig.findobj(Text)if (t.get_gid()or"").startswith(("tag_","title_","panel_title_"))}
    readback=capture_readback(fig,NUMBER,titles,dict(journal=JOURNAL,metadata=metadata or{},
                               pdf_font_sizes=font_sizes,labels=labels,pdf_raster_images=0,
                               **(extra_readback or{})))
    write_json(f"{name}_readback.json",readback)
    if source_tables is not None:
        SOURCES.mkdir(parents=True,exist_ok=True)
        write_xlsx(source_tables,SOURCES/f"Figure{NUMBER}_SourceData.xlsx")
        OUTPUT_RECORDS["xlsx"]=(ROOT/f"04_SOURCE_DATA/Figure{NUMBER}/Figure{NUMBER}_SourceData.xlsx",SOURCES/f"Figure{NUMBER}_SourceData.xlsx")
    preview=SOURCES/f"{name}_PDF_2x.png"
    pdf[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(preview)
    OUTPUT_RECORDS[preview.name]=(ROOT/f"04_SOURCE_DATA/Figure{NUMBER}"/preview.name,preview)
    pdf.close()
    plt.close(fig)
    return readback


def cli(main,number):
    global JOURNAL,OUTPUT,SOURCES,SUPPLEMENTARY,NUMBER,OVERRIDE,EXTERNAL_RESULT
    from figure_export import InputTrace,write_provenance
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--journal",choices=("science","natcatal"),default=os.environ.get("P2_FIGURE_JOURNAL","science"))
    parser.add_argument("--external-result",type=Path,default=os.environ.get("P2_EXTERNAL_RESULT"))
    args=parser.parse_args()
    NUMBER,JOURNAL,OVERRIDE,EXTERNAL_RESULT=number,args.journal,args.out,args.external_result
    if args.out:
        OUTPUT=SOURCES=SUPPLEMENTARY=args.out.resolve()
    elif JOURNAL=="natcatal":
        base=ROOT/"02_FIGURES/Journal_Formats/natcatal"
        OUTPUT,SOURCES,SUPPLEMENTARY=base/"Main",base/f"Figure{number}",base/"Supplementary_Figures"
    else:
        OUTPUT,SOURCES,SUPPLEMENTARY=ROOT/"02_FIGURES/Main",ROOT/f"04_SOURCE_DATA/Figure{number}",ROOT/"02_FIGURES/Supplementary_Figures"
    for path in(OUTPUT,SOURCES,SUPPLEMENTARY):path.mkdir(parents=True,exist_ok=True)
    with InputTrace()as trace:
        result=main()
    helpers={Path(module.__file__).resolve() for module in tuple(sys.modules.values())
             if getattr(module,"__file__",None) and str(getattr(module,"__file__","")).endswith(".py")
             and (ROOT/"06_CODE") in Path(module.__file__).resolve().parents}
    inputs=trace.paths|helpers|{Path(__file__),HERE/f"figure{number}.py"}
    panels=(result or{}).get("panels",{})
    write_provenance(number,inputs,actual_dir=OVERRIDE or (SOURCES if JOURNAL=="natcatal"else None),
                     outputs=OUTPUT_RECORDS,panels=panels)
    print(json.dumps(dict(figure=number,journal=JOURNAL,status="PASS",output=str(OUTPUT)),indent=2))

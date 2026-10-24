"""Shared rendering, deterministic Source Data and provenance for the main figures."""
from pathlib import Path
import json
import hashlib
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd

ROOT = R = Path(__file__).resolve().parents[2]
OUT = ROOT / '02_FIGURES/Main'
SOURCES = ROOT / '04_SOURCE_DATA'
TEMP = ROOT / '08_REVIEW_AND_QC/main_figures'
ACTIVE_OUT = None
ACTIVE_NUMBER = None
DRAWING = None
TABLES = None
MM = 1 / 25.4
PALETTE = dict(halide='#0072B2', amine='#D55E00', acid='#CC79A7',
               other='#009E73', owner='#30343B', partner='#8A9098')
ARRAY_LABELS = {
    'Acylation|1': 'Amide coupling',
    'Buchwald-Hartwig|1': 'C–N (Pd ligand)',
    'Buchwald-Hartwig|2': 'C–N (Pd/Cu/Ni)',
    'C-O competitive reactions|1': 'C–O (Pd/Cu)',
    'Chan-Lam|1': 'Chan–Lam (Cu)',
    'Esterification|1': 'Esterification',
    'Grignard|1': 'Grignard addition',
    'Heck|1': 'Heck (solvent/base)',
    'Hiyama|1': 'Hiyama (Pd)',
    'Kumada|1': 'Kumada (Pd/Ni/Fe)',
    'Reductive-amination|1': 'Reductive amination',
    'SNAr|1': 'SNAr (base/solvent)',
    'Sonogashira|1': 'Sonogashira (solvent/base)',
    'Suzuki|1': 'Suzuki (Pd)',
    'Ullmann|1': 'Ullmann C–O (Cu)',
}
assert len(ARRAY_LABELS) == len(set(ARRAY_LABELS.values())) == 15


def array_label(key, mathtext=True):
    """Return the shared array name, with an optional display-only N subscript."""
    value = ARRAY_LABELS[key]
    return value.replace('SNAr', r'S$_{\mathrm{N}}$Ar') if mathtext else value


def setup_style():
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 9,
        'axes.titleweight': 'normal', 'axes.labelsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.linewidth': 0.6, 'lines.linewidth': 0.8,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Arial',
        'mathtext.it': 'Arial:italic', 'mathtext.bf': 'Arial:bold',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'axes.spines.top': False,
        'axes.spines.right': False, 'text.color': '#222222',
        'axes.labelcolor': '#222222', 'xtick.color': '#222222',
        'ytick.color': '#222222',
    })


def pale(color, factor=0.68):
    rgb = np.asarray(colors.to_rgb(color))
    return colors.to_hex(rgb * (1 - factor) + factor)


def _number(number):
    return str(number).replace('Figure', '').replace('figure', '')


def save_figure(fig, number):
    global DRAWING
    dest = ACTIVE_OUT or OUT
    dest.mkdir(parents=True, exist_ok=True)
    tag = 'Figure' + _number(number)
    assert abs(fig.get_figwidth() / MM - 180) < 0.05, fig.get_size_inches()
    # Round the physical width by 0.0013 mm to obtain exactly 4,252 pixels at 600 dpi.
    fig.set_size_inches(4252 / 600, fig.get_figheight(), forward=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            if artist.axes is not None and not artist.axes.axison:
                tick_text = [label for axis in (artist.axes.xaxis, artist.axes.yaxis)
                             for tick in axis.get_major_ticks() + axis.get_minor_ticks()
                             for label in (tick.label1, tick.label2)]
                if artist in tick_text:
                    continue
            b = artist.get_window_extent(renderer)
            texts.append(dict(text=artist.get_text(), fontsize=artist.get_fontsize(),
                              bbox=list(b.bounds), gid=artist.get_gid(),
                              axes_gid=artist.axes.get_gid() if artist.axes is not None else None))
    panels = [ax.get_title(loc='left') for ax in fig.axes if ax.get_title(loc='left')]
    axes = []
    for ax in fig.axes:
        axes.append(dict(gid=ax.get_gid(), title=ax.get_title(loc='left'), xlim=ax.get_xlim(), ylim=ax.get_ylim(),
            lines=[dict(gid=a.get_gid(), x=np.asarray(a.get_xdata()), y=np.asarray(a.get_ydata()),
                        marker=a.get_marker(), color=colors.to_hex(a.get_color()), mfc=a.get_markerfacecolor())
                   for a in ax.lines],
            collections=[dict(gid=a.get_gid(), offsets=a.get_offsets(), array=a.get_array(), facecolors=a.get_facecolors())
                         for a in ax.collections],
            patches=[dict(gid=a.get_gid(), xy=a.get_xy(), width=a.get_width(), height=a.get_height(), facecolor=a.get_facecolor())
                     for a in ax.patches if isinstance(a, matplotlib.patches.Rectangle)],
            images=[dict(gid=a.get_gid(), array=a.get_array(), extent=a.get_extent(), clim=a.get_clim()) for a in ax.images]))
    DRAWING = clean_json(dict(
        width_mm=fig.get_figwidth()/MM, height_mm=fig.get_figheight()/MM,
        width_px=fig.bbox.width, height_px=fig.bbox.height,
        panel_titles=panels, texts=texts, axes=axes))
    paths = []
    for suffix in ('pdf', 'png', 'svg'):
        path = dest / f'{tag}.{suffix}'
        meta = {'Creator': None, 'Producer': None, 'CreationDate': None} if suffix == 'pdf' else {'Creator': None, 'Date': None} if suffix == 'svg' else {'Software': None}
        fig.savefig(path, dpi=600, bbox_inches=None, pad_inches=0, metadata=meta)
        paths.append(path)
    return paths


def _table(frame):
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    def value(x):
        if isinstance(x, (dict, list, tuple, np.ndarray)):
            return json.dumps(x.tolist() if isinstance(x, np.ndarray) else x, ensure_ascii=False)
        if pd.isna(x):
            return None
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, (str, int, float, bool)) or x is None:
            return x
        return str(x)
    return [[str(c) for c in frame.columns]] + [[value(x) for x in row] for row in frame.itertuples(index=False, name=None)]


def write_sources(number, tables, numbers):
    """Use the existing deterministic scientific workbook writer in the release runtime."""
    global TABLES
    from figure1_data import write_xlsx
    dest = ACTIVE_OUT or SOURCES / f'Figure{number}'
    dest.mkdir(parents=True, exist_ok=True)
    tag = 'Figure' + _number(number)
    TABLES = {'Numbers': numbers, **tables}
    if int(number) == 4:
        rec = tables['D_depth_records']
        TABLES['Fig4a_records'] = rec
        TABLES['Fig4a'] = pd.DataFrame([dict(side=side, k=k, records=len(g),
            additive_profile_explained_pct=100*(1-g[f'sse_{k}'].sum()/g.sst.sum()))
            for side,g in rec.groupby('side') for k in (1,2,3,4,6)])
        tr = tables['C_original_internal']
        TABLES['Fig4b_targets'] = tr
        TABLES['Fig4b'] = pd.DataFrame([dict(row_type='pooled', targets=len(tr), arrays=tr.screen.nunique(),
            owner_fraction=1-tr.sse_owner.sum()/tr.sst.sum(), partner_fraction=1-tr.sse_partner.sum()/tr.sst.sum(),
            D=(tr.sse_partner.sum()-tr.sse_owner.sum())/tr.sst.sum())])
    output = dest / f'{tag}_SourceData.xlsx'
    prepared = {}
    for name,frame in TABLES.items():
        frame = pd.DataFrame(frame).copy()
        for column in frame:
            if frame[column].dtype == object:
                frame[column] = frame[column].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list,tuple)) else v)
        prepared[name] = frame
    write_xlsx(prepared, output)
    return output


def clean_json(value):
    if isinstance(value, np.ma.MaskedArray):
        return clean_json(value.filled(np.nan).tolist())
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run_generator(number, render, out=None):
    global ACTIVE_OUT, ACTIVE_NUMBER
    ACTIVE_OUT = Path(out).resolve() if out else None
    ACTIVE_NUMBER = number
    opened = set()
    capture = [True]
    def hook(event,args):
        if capture[0] and event == 'open' and args and isinstance(args[0], (str,bytes,os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            if ROOT in p.parents and p.is_file() and (any(part in p.parts for part in ('05_DATA','ChemDraw','07_METHODS_AND_PROVENANCE','10_PUBLIC_RELEASE_CANDIDATE')) or p.name in ('Fig4a_folds.csv','Fig4a_history_draws.csv')):
                opened.add(p)
    sys.addaudithook(hook)
    with plt.rc_context({'svg.hashsalt': f'figure{number}-v2'}):
        setup_style()
        render_result = render()
    if number == 2:
        from figure1_data import public_units, noise_scale
        a1,a3 = public_units()
        agreement,scale,n,pearson = noise_scale(a1,a3)
        noise = dict(duplicate_agreement=agreement, model_scale=scale, pairs=n, duplicate_pattern_pearson=pearson)
        panels = dict(A=dict(noise_calibration=noise))
    else:
        panels = {}
    capture[0] = False
    tag = f'Figure{number}'
    source_dir = SOURCES / tag
    qc = ROOT / f'08_REVIEW_AND_QC/figure{number}'
    actual_dir = ACTIVE_OUT or source_dir
    actual_dir.mkdir(parents=True, exist_ok=True)
    (ACTIVE_OUT or qc).mkdir(parents=True, exist_ok=True)
    logical = dict(pdf=OUT/f'{tag}.pdf', svg=OUT/f'{tag}.svg', png=OUT/f'{tag}.png',
                   xlsx=source_dir/f'{tag}_SourceData.xlsx', readback=qc/f'figure{number}_plot_readback.json')
    actual = {k: (ACTIVE_OUT / p.name if ACTIVE_OUT else p) for k,p in logical.items()}
    if number == 4:
        assert render_result and len(render_result) == 2, "Figure 4 must regenerate both Set A bootstrap tables"
        for key, paths in render_result.items():
            logical[key], actual[key] = paths['logical'], paths['actual']
    actual['readback'].write_text(json.dumps(DRAWING, indent=2, ensure_ascii=False), encoding='utf-8')
    if number == 4:
        for name in ('Fig4a_folds.csv','Fig4a_history_draws.csv'):
            p = source_dir/name
            logical[name] = p
            actual[name] = ACTIVE_OUT/name if ACTIVE_OUT else p
            if actual[name] != p:
                actual[name].write_bytes(p.read_bytes())
    rel = lambda p: p.relative_to(ROOT).as_posix()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    prov = dict(figure=f'Figure {number}', generated_by=f'06_CODE/Figures/figure{number}.py',
        inputs={str(i):dict(file=rel(p),sha256=sha(p)) for i,p in enumerate(sorted(opened))},
        panels=panels,
        outputs={k:dict(file=rel(logical[k]),sha256=sha(p)) for k,p in actual.items()})
    (actual_dir/f'{tag}_SourceData_provenance.json').write_text(json.dumps(prov,indent=1,ensure_ascii=False),encoding='utf-8')
    return prov

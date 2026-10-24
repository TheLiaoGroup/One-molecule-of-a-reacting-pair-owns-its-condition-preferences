"""Deterministic figure records shared by the formal generators."""
from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT_ROLES = {
    "05_DATA/Analysis/06_METADATA/frozen_folds.json": "fold_registry",
    "05_DATA/Analysis/06_METADATA/input_lock.json": "registry_integrity",
    "07_METHODS_AND_PROVENANCE/analysis_runtime/PATH_RELOCATIONS.json": "registry_integrity",
    "06_CODE/Directional_Ownership/phase2_section7.py": "phase2_code",
    "05_DATA/Results/validation/matched/fold_assignments.csv": "frozen",
    **{f"05_DATA/Analysis/figure_display/{name}.csv": "released"
       for name in ("Fig4a_folds", "Fig4a_history_draws")},
}


def clean(value):
    if isinstance(value, np.ma.MaskedArray):
        return clean(value.filled(np.nan).tolist())
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class InputTrace(AbstractContextManager):
    """Observe scientific file reads; the hook is inactive outside this context."""

    def __init__(self):
        self.paths = set()
        self.active = False
        sys.addaudithook(self._hook)

    def _hook(self, event, args):
        if not self.active or event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ""
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if any(c in mode for c in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if ROOT in path.parents and path.is_file() and (
            path.relative_to(ROOT).as_posix() in INPUT_ROLES or any(
                part in path.relative_to(ROOT).parts for part in
                ("05_DATA", "04_SOURCE_DATA", "ChemDraw", "07_METHODS_AND_PROVENANCE", "10_PUBLIC_RELEASE_CANDIDATE")
            )
        ):
            self.paths.add(path)

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *args):
        self.active = False


def capture_readback(fig, number, titles, extra=None):
    from matplotlib.collections import PathCollection, QuadMesh
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        if artist.axes is not None and not artist.axes.axison:
            ticks = [label for axis in (artist.axes.xaxis, artist.axes.yaxis)
                     for tick in axis.get_major_ticks() + axis.get_minor_ticks()
                     for label in (tick.label1, tick.label2)]
            if artist in ticks:
                continue
        texts.append(dict(text=artist.get_text(), fontsize=artist.get_fontsize(),
                          family=artist.get_fontfamily(), bbox=list(artist.get_window_extent(renderer).bounds),
                          gid=artist.get_gid(), axes_gid=artist.axes.get_gid() if artist.axes else None))
    axes = []
    for ax in fig.axes:
        axes.append(dict(gid=ax.get_gid(), xlim=ax.get_xlim(), ylim=ax.get_ylim(),
                         bbox=list(ax.get_window_extent(renderer).bounds),
                         lines=[dict(gid=a.get_gid(), x=a.get_xdata(), y=a.get_ydata(),
                                     marker=a.get_marker(), color=a.get_color(),
                                     markersize=a.get_markersize(), linestyle=a.get_linestyle())
                                for a in ax.lines],
                         meshes=[dict(gid=a.get_gid(), array=a.get_array(),
                                      coordinates=a.get_coordinates(), clim=a.get_clim())
                                 for a in ax.collections if isinstance(a, QuadMesh)],
                         collections=[dict(gid=a.get_gid(), offsets=a.get_offsets(), array=a.get_array(),
                                           sizes=a.get_sizes(), facecolors=a.get_facecolors())
                                      for a in ax.collections if isinstance(a, PathCollection)],
                         patches=[dict(gid=a.get_gid(), xy=a.get_xy(), width=a.get_width(), height=a.get_height())
                                  for a in ax.patches if isinstance(a, Rectangle)],
                         images=[dict(gid=a.get_gid(), array=a.get_array(), extent=a.get_extent(), clim=a.get_clim())
                                 for a in ax.images]))
    return clean(dict(figure=number, width_mm=fig.get_figwidth() * 25.4,
                      height_mm=fig.get_figheight() * 25.4, width_px=fig.bbox.width,
                      height_px=fig.bbox.height, dpi=fig.dpi,
                      panel_titles=titles, texts=texts, axes=axes, **(extra or {})))


def write_provenance(number, inputs, actual_dir=None, outputs=None, panels=None):
    """Hash actual outputs while retaining their stable logical release paths.

    outputs maps a record key to either a logical Path or (logical, actual).
    actual_dir redirects bare logical paths by filename for isolated builds.
    """
    number = int(number)
    source = ROOT / f"04_SOURCE_DATA/Figure{number}"
    logical = {ext: ROOT / f"02_FIGURES/Main/Figure{number}.{ext}" for ext in ("svg", "pdf", "png")}
    logical.update(xlsx=source / f"Figure{number}_SourceData.xlsx",
                   readback=source / f"Figure{number}_readback.json")
    logical.update(outputs or {})
    records, own_outputs = {}, set()
    for key, item in logical.items():
        if isinstance(item, (tuple, list)):
            target, actual = map(Path, item)
        else:
            target = Path(item)
            actual = Path(actual_dir) / target.name if actual_dir else target
        records[key] = dict(file=target.relative_to(ROOT).as_posix(),
                            sha256=hashlib.sha256(actual.read_bytes()).hexdigest())
        own_outputs.add(actual.resolve())
    input_paths = sorted({Path(p).resolve() for p in inputs} - own_outputs)
    input_records = {}
    for path in input_paths:
        relative = path.relative_to(ROOT).as_posix()
        role = INPUT_ROLES.get(relative, "files")
        input_records.setdefault(role, {})[relative] = dict(
            file=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result = dict(figure=f"Figure {number}", generated_by=f"06_CODE/Figures/figure{number}.py",
                  inputs=input_records, outputs=records, panels=panels or {})
    dest = Path(actual_dir) if actual_dir else source
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"Figure{number}_SourceData_provenance.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return result

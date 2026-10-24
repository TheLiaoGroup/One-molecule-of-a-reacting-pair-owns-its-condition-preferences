"""Rendered D21 geometry; extracted unchanged from approved draft QA.

The caller owns evidence persistence. No file is read or written here.
"""
from __future__ import annotations

import math
import re

from matplotlib.collections import PathCollection
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path as MplPath
from matplotlib.text import Annotation, Text
import numpy as np

MIN_CLEARANCE_PT = 2.0


def canonical(text):
    text = str(text).replace("\u2212", "-").replace("\n", " ")
    return re.sub(r"\$_\{(\d+)\}\$", r"\1", text)


def native_gid(gid):
    return "|label" in (gid or "")


def text_gap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return math.hypot(max(ax - bx - bw, bx - ax - aw, 0),
                      max(ay - by - bh, by - ay - ah, 0))


def _point_rect(points, box):
    low, high = np.asarray(box[:2]), np.asarray(box[:2]) + box[2:]
    return np.linalg.norm(np.maximum(np.maximum(low - points, points - high), 0), axis=1)


def _segments_rect(segments, box):
    """Exact Euclidean distance from line segments to an axis-aligned text box."""
    a, b = segments[:, 0], segments[:, 1]
    delta = b - a
    low, high = np.asarray(box[:2]), np.asarray(box[:2]) + box[2:]
    corners = np.asarray([[low[0], low[1]], [low[0], high[1]],
                          [high[0], low[1]], [high[0], high[1]]])
    denom = np.sum(delta * delta, axis=1)
    ratios = np.divide(np.sum((corners[:, None] - a) * delta, axis=2), denom,
                       out=np.zeros((4, len(a))), where=denom > 0)
    closest = a + np.clip(ratios, 0, 1)[..., None] * delta
    distance = np.minimum(_point_rect(a, box), _point_rect(b, box))
    distance = np.minimum(distance, np.linalg.norm(corners[:, None] - closest, axis=2).min(axis=0))
    enter, leave = np.zeros(len(a)), np.ones(len(a))
    intersects = np.ones(len(a), dtype=bool)
    for axis in range(2):
        moving = np.abs(delta[:, axis]) > 1e-12
        ratios0 = np.divide(low[axis] - a[:, axis], delta[:, axis],
                            out=np.zeros(len(a)), where=moving)
        ratios1 = np.divide(high[axis] - a[:, axis], delta[:, axis],
                            out=np.ones(len(a)), where=moving)
        enter = np.maximum(enter, np.where(moving, np.minimum(ratios0, ratios1), 0))
        leave = np.minimum(leave, np.where(moving, np.maximum(ratios0, ratios1), 1))
        intersects &= moving | ((a[:, axis] >= low[axis]) & (a[:, axis] <= high[axis]))
    distance[intersects & (enter <= leave)] = 0
    return distance


def _segments(path):
    segments, current, start = [], None, None
    for values, code in path.iter_segments(curves=False, simplify=False):
        point = np.asarray(values[-2:])
        if code == MplPath.MOVETO:
            current = start = point
        elif code == MplPath.CLOSEPOLY:
            if current is not None and start is not None:
                segments.append((current, start))
            current = start
        elif current is not None:
            segments.append((current, point))
            current = point
    return np.asarray(segments, dtype=float).reshape(-1, 2, 2)


def _clip_segments(segments, bounds):
    if not len(segments):
        return segments
    low, high = bounds.get_points()
    a, b = segments[:, 0], segments[:, 1]
    delta = b - a
    enter, leave = np.zeros(len(a)), np.ones(len(a))
    keep = np.ones(len(a), dtype=bool)
    for axis in range(2):
        moving = np.abs(delta[:, axis]) > 1e-12
        t0 = np.divide(low[axis] - a[:, axis], delta[:, axis], out=np.zeros(len(a)), where=moving)
        t1 = np.divide(high[axis] - a[:, axis], delta[:, axis], out=np.ones(len(a)), where=moving)
        enter = np.maximum(enter, np.where(moving, np.minimum(t0, t1), 0))
        leave = np.minimum(leave, np.where(moving, np.maximum(t0, t1), 1))
        keep &= moving | ((a[:, axis] >= low[axis]) & (a[:, axis] <= high[axis]))
    keep &= enter <= leave
    return np.stack((a + enter[:, None] * delta, a + leave[:, None] * delta), axis=1)[keep]


def d21(fig, number):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    ticks = {label for ax in fig.axes for axis in (ax.xaxis, ax.yaxis)
             for tick in axis.get_major_ticks() + axis.get_minor_ticks()
             for label in (tick.label1, tick.label2)}
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()
             and not (t in ticks and t.axes is not None and not t.axes.axison)]
    # Annotation.get_window_extent unions the leader with its label.
    boxes = {id(t): tuple(Text.get_window_extent(t, renderer).bounds) for t in texts}
    bbox_corrections = [dict(text=t.get_text(), gid=t.get_gid(), text_bbox=list(boxes[id(t)]),
                             artist_bbox=list(t.get_window_extent(renderer).bounds))
                        for t in texts if isinstance(t, Annotation)
                        and boxes[id(t)] != tuple(t.get_window_extent(renderer).bounds)]
    annotations = [t for t in texts if t not in ticks and not native_gid(t.get_gid())
                   and not (t.get_gid() or "").startswith(("panel_", "tag_"))]
    arrow_owners = {id(t.arrow_patch): t for t in texts
                    if isinstance(t, Annotation) and t.arrow_patch is not None}
    geometries = []

    def add(artist, segments, radius=0, kind="line", center=None):
        if artist.get_clip_on() and artist.axes is not None:
            segments = _clip_segments(segments, artist.axes.bbox)
        if len(segments):
            geometries.append(dict(artist=artist, gid=artist.get_gid(), kind=kind,
                                   segments=segments, radius=radius, center=center))

    for ax in fig.axes:
        for line in ax.lines:
            if not line.get_visible():
                continue
            path = line.get_transform().transform_path(line.get_path())
            if line.get_linestyle() not in ("None", "", " ", None) and line.get_linewidth() > 0:
                add(line, _segments(path), line.get_linewidth() * px_per_pt / 2)
            if line.get_marker() not in ("None", "", " ", None):
                points = path.vertices[np.isfinite(path.vertices).all(axis=1)]
                if line.get_clip_on() and len(points):
                    points = points[(points[:, 0] >= ax.bbox.x0) & (points[:, 0] <= ax.bbox.x1)
                                    & (points[:, 1] >= ax.bbox.y0) & (points[:, 1] <= ax.bbox.y1)]
                marker = MarkerStyle(line.get_marker())
                marker_segments = _segments(marker.get_transform().transform_path(marker.get_path()))
                radius = line.get_markeredgewidth() * px_per_pt / 2
                for point in points:
                    add(line, marker_segments * line.get_markersize() * px_per_pt + point,
                        radius, "marker", point)
        for collection in ax.collections:
            if not isinstance(collection, PathCollection) or not collection.get_visible():
                continue
            offsets = np.asarray(collection.get_offset_transform().transform(collection.get_offsets()))
            sizes, widths = collection.get_sizes(), collection.get_linewidths()
            for index, point in enumerate(offsets):
                if not np.isfinite(point).all() or (collection.get_clip_on() and not ax.bbox.contains(*point)):
                    continue
                path = collection.get_paths()[index % len(collection.get_paths())]
                segments = _segments(path) * math.sqrt(sizes[index % len(sizes)]) * px_per_pt + point
                radius = widths[index % len(widths)] * px_per_pt / 2
                add(collection, segments, radius, "scatter", point)
        for patch in ax.patches:
            if patch.get_visible():
                add(patch, _segments(patch.get_transform().transform_path(patch.get_path())),
                    patch.get_linewidth() * px_per_pt / 2, "patch")
        if ax.axison:
            for spine in ax.spines.values():
                if spine.get_visible():
                    add(spine, _segments(spine.get_transform().transform_path(spine.get_path())),
                        spine.get_linewidth() * px_per_pt / 2, "axis spine")
    for t in texts:
        if isinstance(t, Annotation) and t.arrow_patch is not None and t.arrow_patch.get_visible():
            arrow = t.arrow_patch
            add(arrow, _segments(arrow.get_transform().transform_path(arrow.get_path())),
                arrow.get_linewidth() * px_per_pt / 2, "connected leader")
    rows, exceptions = [], []
    for text in annotations:
        box = boxes[id(text)]
        nearest_graphic, graphic_distance = None, float("inf")
        for geometry in geometries:
            if arrow_owners.get(id(geometry["artist"])) is text:
                exceptions.append(dict(text=text.get_text(), gid=text.get_gid(),
                                       reason="This annotation's own leader attaches to its label; its data endpoint is still measured."))
                continue
            distance = max(0, float(_segments_rect(geometry["segments"], box).min()) - geometry["radius"])
            if distance < graphic_distance:
                graphic_distance = distance
                nearest_graphic = dict(gid=geometry["gid"], kind=geometry["kind"])
        neighbors = [(text_gap(box, boxes[id(other)]), other) for other in texts if other is not text]
        gap, other = min(neighbors, key=lambda item: item[0]) if neighbors else (float("inf"), None)
        rows.append(dict(text=text.get_text(), gid=text.get_gid(),
                         graphic_gap_pt=graphic_distance / px_per_pt if np.isfinite(graphic_distance) else None,
                         nearest_graphic=nearest_graphic,
                         text_gap_pt=gap / px_per_pt if np.isfinite(gap) else None,
                         nearest_text=other.get_text() if other is not None else None,
                         nearest_text_gid=other.get_gid() if other is not None else None))
    failures = [row for row in rows if any(row[key] is not None and row[key] < MIN_CLEARANCE_PT - 1e-6
                                         for key in ("graphic_gap_pt", "text_gap_pt"))]
    result = dict(status="FAIL" if failures else "PASS", figure=number,
                  measurement_version=3,
                  criterion_pt=MIN_CLEARANCE_PT, measurement_dpi=fig.dpi,
                  method="Rendered text bboxes versus transformed visible marker edges, line strokes, patch paths, and all other visible text bboxes; Euclidean minimum clearance in print points.",
                  scope="All ordinary non-tick text except panel letters and panel tags; tick labels, axis titles and native chemical labels remain text obstacles.",
                  annotations=len(rows), geometry_primitives=len(geometries),
                  minimum_graphic_gap_pt=min((r["graphic_gap_pt"] for r in rows if r["graphic_gap_pt"] is not None), default=None),
                  minimum_text_gap_pt=min((r["text_gap_pt"] for r in rows if r["text_gap_pt"] is not None), default=None),
                  annotation_text_bbox_corrections=bbox_corrections,
                  connected_leader_exceptions=exceptions, rows=rows, failures=failures)
    return result

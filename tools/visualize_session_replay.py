#!/usr/bin/env python3
"""
Interactive session replay: frame-by-frame YOLO segmentation + depth map viewer.

Run after a WebRTC session closes. Joins edge_session.jsonl with NPZ artifacts
by trace_id. Frames missing an NPZ are silently skipped.

Controls:
  Any key   → advance one frame
  q / ESC   → quit
  Trackbars → adjust near/far depth thresholds live (recalors masks instantly)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import orjson

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: opencv-python.  "
        "pip install -r tools/requirements.txt"
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: numpy.  "
        "pip install -r tools/requirements.txt"
    ) from exc


# ── window / UI constants ───────────────────────────────────────────────────

WIN_YOLO  = "YOLO Segmentation"
WIN_DEPTH = "Depth Map"
WIN_CTRL  = "Controls"

# BGR colours for each depth band
BAND_COLOR: dict[str, tuple[int, int, int]] = {
    "near": (0,   60, 220),   # red
    "mid":  (0,  200, 220),   # yellow
    "far":  (60, 200,  60),   # green
}

DEFAULT_NEAR_PCT = 35   # trackbar default (0-100 → 0.0-1.0)
DEFAULT_FAR_PCT  = 65


# ── helpers ─────────────────────────────────────────────────────────────────

def _num(v: Any, default: float = 0.0) -> float:
    return float(v) if isinstance(v, (int, float)) else default


def _text(
    img: np.ndarray,
    txt: str,
    x: int,
    y: int,
    scale: float = 0.50,
    color: tuple[int, int, int] = (220, 220, 220),
    thickness: int = 1,
) -> None:
    cv2.putText(
        img, txt, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
    )


def _stale_border(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 220), 6)


def _band_from_norm(depth_norm: float, near_t: float, far_t: float) -> str:
    if depth_norm < near_t:
        return "near"
    if depth_norm > far_t:
        return "far"
    return "mid"


# ── data loading ─────────────────────────────────────────────────────────────

def _load_inference_rows(path: Path) -> list[dict[str, Any]]:
    """Read every inference_done row from a JSONL file, sorted by inference_ts_ms."""
    rows: list[dict[str, Any]] = []
    with path.open("rb") as fh:
        for line in fh:
            try:
                obj = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("event") == "inference_done":
                rows.append(obj)
    rows.sort(key=lambda r: int(_num(r.get("inference_ts_ms"), 0)))
    return rows


def _load_artifact(artifact_dir: Path, trace_id: str) -> dict[str, Any] | None:
    path = artifact_dir / f"{trace_id}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _overlap_objects(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the fused overlap object list from an inference_done row."""
    detections = row.get("detections")
    if not isinstance(detections, dict):
        return []
    overlap = detections.get("overlap")
    if not isinstance(overlap, dict):
        return []
    objs = overlap.get("objects")
    return objs if isinstance(objs, list) else []


def _depth_percentiles(row: dict[str, Any]) -> dict[str, float]:
    detections = row.get("detections") or {}
    depth_info  = detections.get("depth") or {}
    pcts        = depth_info.get("depth_percentiles") or {}
    return {
        "p10":    _num(pcts.get("p10")),
        "p50":    _num(pcts.get("p50")),
        "p90":    _num(pcts.get("p90")),
        "spread": _num(pcts.get("spread_p90_p10")),
    }


# ── YOLO window render ────────────────────────────────────────────────────────

def render_yolo(
    artifact:       dict[str, Any],
    overlap_objs:   list[dict[str, Any]],
    is_stale:       bool,
    near_t:         float,
    far_t:          float,
    frame_idx:      int,
    total:          int,
) -> np.ndarray:
    """
    Black canvas at mask native resolution.
    Masks coloured by live depth band (slider-driven).
    Labels: class_name | conf | band | depth_median
    """
    masks:       np.ndarray | None = artifact.get("yolo_masks")
    class_names: np.ndarray | None = artifact.get("class_names")
    confidences: np.ndarray | None = artifact.get("confidences")
    bboxes:      np.ndarray | None = artifact.get("bboxes")
    frame_bgr:   np.ndarray | None = artifact.get("frame_bgr")

    # determine canvas size from mask or frame dims
    if isinstance(masks, np.ndarray) and masks.ndim == 3 and masks.shape[0] > 0:
        H, W = int(masks.shape[1]), int(masks.shape[2])
    elif isinstance(frame_bgr, np.ndarray) and frame_bgr.ndim == 3:
        H, W = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    else:
        H, W = 480, 640

    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # per-object depth_norm from JSONL (drives live band recolouring)
    depth_norms: list[float] = [
        _num(o.get("depth_norm"), 0.5) for o in overlap_objs
    ]

    # draw filled masks
    if isinstance(masks, np.ndarray) and masks.ndim == 3:
        for i in range(masks.shape[0]):
            dn   = depth_norms[i] if i < len(depth_norms) else 0.5
            band = _band_from_norm(dn, near_t, far_t)
            canvas[masks[i] > 0] = BAND_COLOR[band]

    # draw bounding boxes + labels
    if isinstance(bboxes, np.ndarray) and bboxes.ndim == 2 and bboxes.shape[1] == 4:
        n = bboxes.shape[0]
        for i in range(n):
            x1, y1, x2, y2 = (int(float(v)) for v in bboxes[i])
            dn   = depth_norms[i] if i < len(depth_norms) else 0.5
            band = _band_from_norm(dn, near_t, far_t)
            color = BAND_COLOR[band]

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            cls  = str(class_names[i]) if isinstance(class_names, np.ndarray) and i < len(class_names) else "obj"
            conf = float(confidences[i]) if isinstance(confidences, np.ndarray) and i < len(confidences) else 0.0
            d_med = _num(overlap_objs[i].get("depth_median")) if i < len(overlap_objs) else 0.0
            label = f"{cls}  {conf:.2f}  {band}  d={d_med:.3f}"
            _text(canvas, label, x1 + 2, max(14, y1 - 5), scale=0.46, color=color, thickness=1)

    # HUD: frame counter + active thresholds
    _text(canvas, f"frame {frame_idx + 1}/{total}", 6, 16, scale=0.48, color=(200, 200, 200))
    _text(canvas, f"near < {near_t:.2f}  |  far > {far_t:.2f}", 6, H - 8, scale=0.44, color=(160, 160, 160))

    if is_stale:
        _stale_border(canvas)
        _text(canvas, "STALE", W - 72, 22, scale=0.55, color=(0, 0, 220), thickness=2)

    return canvas


# ── Depth window render ───────────────────────────────────────────────────────

def render_depth(
    artifact:  dict[str, Any],
    row:       dict[str, Any],
    is_stale:  bool,
    near_t:    float,
    far_t:     float,
    frame_idx: int,
    total:     int,
) -> np.ndarray:
    """
    COLORMAP_VIRIDIS depth map.
    Overlay: p10/p50/p90 text + colorbar with near/far threshold lines.
    """
    depth_map: np.ndarray | None = artifact.get("depth_map")

    if not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
        panel = np.zeros((480, 640, 3), dtype=np.uint8)
        _text(panel, "No depth map in artifact", 20, 240, scale=0.65, color=(180, 180, 180))
        return panel

    depth = depth_map.astype(np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        panel = np.zeros((480, 640, 3), dtype=np.uint8)
        _text(panel, "Depth map contains no finite values", 20, 240, scale=0.55)
        return panel

    # normalise to [1%, 99%] to suppress outliers
    d_min = float(np.percentile(depth[finite], 1.0))
    d_max = float(np.percentile(depth[finite], 99.0))
    if d_max <= d_min:
        d_max = d_min + 1e-6

    norm     = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
    depth_u8 = (norm * 255.0).astype(np.uint8)
    panel    = cv2.applyColorMap(depth_u8, cv2.COLORMAP_VIRIDIS)
    panel    = cv2.resize(panel, (640, 480), interpolation=cv2.INTER_LINEAR)

    # percentiles from JSONL metadata
    pcts = _depth_percentiles(row)
    _text(panel,
          f"p10={pcts['p10']:.3f}   p50={pcts['p50']:.3f}   p90={pcts['p90']:.3f}   spread={pcts['spread']:.3f}",
          8, 20, scale=0.48, color=(255, 255, 255))
    _text(panel, f"raw [{d_min:.3f}, {d_max:.3f}]", 8, 40, scale=0.42, color=(200, 200, 200))
    _text(panel, f"frame {frame_idx + 1}/{total}", 8, 468, scale=0.44, color=(200, 200, 200))

    # ── colorbar with threshold lines ──────────────────────────────────────
    bar_x          = 608
    bar_y0, bar_y1 = 60, 420
    bar_h          = bar_y1 - bar_y0
    bar_w          = 18

    for yy in range(bar_h):
        t   = 1.0 - (yy / bar_h)          # top = high intensity
        val = int(t * 255)
        col = cv2.applyColorMap(np.array([[val]], dtype=np.uint8), cv2.COLORMAP_VIRIDIS)[0, 0].tolist()
        cv2.line(panel, (bar_x, bar_y0 + yy), (bar_x + bar_w, bar_y0 + yy), col, 1)

    # near threshold line (low depth_norm = near = low bar position)
    near_y = bar_y1 - int(near_t * bar_h)
    far_y  = bar_y1 - int(far_t  * bar_h)
    cv2.line(panel, (bar_x - 5, near_y), (bar_x + bar_w + 5, near_y), (0, 60, 220), 2)
    cv2.line(panel, (bar_x - 5, far_y),  (bar_x + bar_w + 5, far_y),  (60, 200, 60), 2)
    _text(panel, "near", bar_x - 42, near_y + 4, scale=0.37, color=(0, 60, 220))
    _text(panel, "far",  bar_x - 34, far_y  + 4, scale=0.37, color=(60, 200, 60))

    if is_stale:
        _stale_border(panel)
        _text(panel, "STALE", 640 - 72, 22, scale=0.55, color=(0, 0, 220), thickness=2)

    return panel


# ── controls window ───────────────────────────────────────────────────────────

def _create_controls_window() -> None:
    ctrl = np.zeros((110, 400, 3), dtype=np.uint8)
    _text(ctrl, "Depth Band Calibration", 10, 24, scale=0.58, color=(255, 255, 255))
    _text(ctrl, "near threshold  (0-100 → 0.0-1.0)", 10, 52, scale=0.42, color=(160, 160, 160))
    _text(ctrl, "far  threshold  (0-100 → 0.0-1.0)", 10, 72, scale=0.42, color=(160, 160, 160))
    _text(ctrl, "any key = next frame   q/ESC = quit", 10, 96, scale=0.40, color=(120, 120, 120))
    cv2.imshow(WIN_CTRL, ctrl)
    cv2.createTrackbar("near (x0.01)", WIN_CTRL, DEFAULT_NEAR_PCT, 100, lambda _: None)
    cv2.createTrackbar("far  (x0.01)", WIN_CTRL, DEFAULT_FAR_PCT,  100, lambda _: None)


def _read_thresholds() -> tuple[float, float]:
    near_t = cv2.getTrackbarPos("near (x0.01)", WIN_CTRL) / 100.0
    far_t  = cv2.getTrackbarPos("far  (x0.01)", WIN_CTRL) / 100.0
    # clamp so near is always below far
    near_t = min(near_t, far_t - 0.01)
    return near_t, far_t


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive frame-by-frame YOLO + Depth session replay."
    )
    parser.add_argument(
        "--edge-input",
        default="logs/edge_session.jsonl",
        help="Path to edge_session.jsonl",
    )
    parser.add_argument(
        "--artifact-dir",
        default="logs/visualization_artifacts",
        help="Directory containing {trace_id}.npz files",
    )
    args = parser.parse_args()

    edge_path    = Path(args.edge_input)
    artifact_dir = Path(args.artifact_dir)

    if not edge_path.exists():
        raise SystemExit(f"JSONL not found: {edge_path}")

    print(f"Loading inference rows from: {edge_path}")
    all_rows = _load_inference_rows(edge_path)
    print(f"  {len(all_rows)} inference_done rows loaded")

    # filter to only rows that have a matching NPZ
    paired: list[dict[str, Any]] = []
    skipped = 0
    for row in all_rows:
        trace_id = row.get("trace_id", "")
        if (artifact_dir / f"{trace_id}.npz").exists():
            paired.append(row)
        else:
            skipped += 1

    print(f"  {len(paired)} frames with NPZ  |  {skipped} skipped (no artifact)")

    if not paired:
        raise SystemExit(
            "No frames with matching NPZ artifacts found.  "
            "Check --artifact-dir and that visualization_dump_enabled=true in your config."
        )

    total = len(paired)

    # open windows
    cv2.namedWindow(WIN_YOLO,  cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_DEPTH, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_CTRL,  cv2.WINDOW_NORMAL)
    _create_controls_window()

    print(f"\nShowing {total} frames.  any key = next frame  |  q/ESC = quit\n")

    for idx, row in enumerate(paired):
        trace_id = row.get("trace_id", "")
        is_stale = bool(row.get("is_stale"))
        artifact = _load_artifact(artifact_dir, trace_id)

        if artifact is None:
            # shouldn't happen after filtering, but guard anyway
            continue

        overlap_objs = _overlap_objects(row)
        near_t, far_t = _read_thresholds()

        yolo_frame  = render_yolo(artifact, overlap_objs, is_stale, near_t, far_t, idx, total)
        depth_frame = render_depth(artifact, row, is_stale, near_t, far_t, idx, total)

        cv2.imshow(WIN_YOLO,  yolo_frame)
        cv2.imshow(WIN_DEPTH, depth_frame)

        stale_tag = "  [STALE]" if is_stale else ""
        n_objs    = len(overlap_objs)
        age_ms    = _num(row.get("age_ms"))
        print(f"  [{idx + 1:>4}/{total}]  trace={trace_id[:12]}  objs={n_objs}  age={age_ms:.0f}ms{stale_tag}")

        # wait indefinitely for a keypress (frame-by-frame advance)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):   # q or ESC
            print("Quit.")
            break

    cv2.destroyAllWindows()
    print("\nReplay complete.")


if __name__ == "__main__":
    main()

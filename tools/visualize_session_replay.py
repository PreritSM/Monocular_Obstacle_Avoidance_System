from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: opencv-python. Install with: "
        "python3 -m pip install -r tools/requirements.txt"
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: numpy. Install with: "
        "python3 -m pip install -r tools/requirements.txt"
    ) from exc


@dataclass
class FrameRecord:
    trace_id: str
    ts_ms: int
    age_ms: float
    stale_threshold_ms: float
    is_stale: bool
    yolo_inference_ms: float
    depth_inference_ms: float
    capture_to_edge_ms: float
    edge_to_done_ms: float
    object_count: int


def _num(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _load_jsonl_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("rb") as f:
        for line in f:
            try:
                obj = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _build_frames(jetson_rows: list[dict[str, Any]], edge_rows: list[dict[str, Any]]) -> list[FrameRecord]:
    by_trace: dict[str, dict[str, Any]] = {}

    for row in edge_rows:
        if row.get("event") != "inference_done":
            continue
        trace_id = row.get("trace_id")
        if isinstance(trace_id, str):
            by_trace.setdefault(trace_id, {})["payload"] = row

    for row in jetson_rows:
        if row.get("event") != "metadata_rx":
            continue
        raw_msg = row.get("message")
        if not isinstance(raw_msg, str):
            continue
        try:
            payload = orjson.loads(raw_msg)
        except orjson.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        trace_id = payload.get("trace_id")
        if isinstance(trace_id, str):
            by_trace.setdefault(trace_id, {})["payload"] = payload

    frames: list[FrameRecord] = []
    for trace_id, slot in by_trace.items():
        payload = slot.get("payload")
        if not isinstance(payload, dict):
            continue

        timings = payload.get("timings_ms") if isinstance(payload.get("timings_ms"), dict) else {}
        yolo_t = timings.get("yolo") if isinstance(timings.get("yolo"), dict) else {}
        depth_t = timings.get("depth") if isinstance(timings.get("depth"), dict) else {}
        detections = payload.get("detections") if isinstance(payload.get("detections"), dict) else {}
        yolo_det = detections.get("yolo") if isinstance(detections.get("yolo"), dict) else {}

        ts_ms = int(_num(payload.get("inference_ts_ms"), 0.0))
        if ts_ms <= 0:
            continue

        frames.append(
            FrameRecord(
                trace_id=trace_id,
                ts_ms=ts_ms,
                age_ms=_num(payload.get("age_ms"), 0.0),
                stale_threshold_ms=_num(payload.get("stale_threshold_ms"), 120.0),
                is_stale=bool(payload.get("is_stale")),
                yolo_inference_ms=_num(yolo_t.get("inference_ms"), 0.0),
                depth_inference_ms=_num(depth_t.get("inference_ms"), 0.0),
                capture_to_edge_ms=_num(timings.get("capture_to_edge_rx_ms"), 0.0),
                edge_to_done_ms=_num(timings.get("edge_rx_to_inference_done_ms"), 0.0),
                object_count=int(_num(yolo_det.get("object_count"), 0.0)),
            )
        )

    frames.sort(key=lambda x: x.ts_ms)
    return frames


def _draw_text(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.58, color: tuple[int, int, int] = (220, 220, 220), thickness: int = 1) -> None:
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)


def _load_artifact(artifact_dir: Path, trace_id: str) -> dict[str, Any] | None:
    path = artifact_dir / f"{trace_id}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    out: dict[str, Any] = {}
    for key in data.files:
        out[key] = data[key]
    return out


def _color_for_index(i: int) -> tuple[int, int, int]:
    palette = [
        (40, 220, 255),
        (255, 120, 50),
        (140, 255, 80),
        (255, 80, 180),
        (80, 180, 255),
        (220, 220, 80),
    ]
    return palette[i % len(palette)]


def _render_yolo_board(artifact: dict[str, Any], width: int, height: int) -> np.ndarray:
    board = np.zeros((height, width, 3), dtype=np.uint8)
    _draw_text(board, "YOLO Segmentation Masks (Blackboard)", 20, 32, scale=0.8, color=(255, 255, 255), thickness=2)

    frame = artifact.get("frame_bgr")
    masks = artifact.get("yolo_masks")
    class_names = artifact.get("class_names")
    confidences = artifact.get("confidences")
    bboxes = artifact.get("bboxes")

    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        thumb_w = min(380, width // 3)
        thumb_h = int(frame.shape[0] * (thumb_w / frame.shape[1]))
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_LINEAR)
        board[50 : 50 + thumb_h, 20 : 20 + thumb_w] = thumb
        _draw_text(board, "camera frame (inset)", 20, 50 + thumb_h + 20, scale=0.5, color=(160, 160, 160))

    if not isinstance(masks, np.ndarray) or masks.ndim != 3 or masks.shape[0] == 0:
        _draw_text(board, "No YOLO masks available for this frame.", 20, height // 2, scale=0.7, color=(180, 180, 180))
        return board

    mh, mw = masks.shape[1], masks.shape[2]
    overlay = np.zeros((mh, mw, 3), dtype=np.uint8)

    for i in range(masks.shape[0]):
        mask = masks[i] > 0
        color = _color_for_index(i)
        overlay[mask] = color

    overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_NEAREST)
    board = cv2.addWeighted(board, 1.0, overlay, 0.55, 0)

    if isinstance(bboxes, np.ndarray) and bboxes.ndim == 2 and bboxes.shape[1] == 4:
        sx = width / float(mw)
        sy = height / float(mh)
        for i in range(min(bboxes.shape[0], masks.shape[0])):
            x1, y1, x2, y2 = [float(v) for v in bboxes[i]]
            p1 = (int(x1 * sx), int(y1 * sy))
            p2 = (int(x2 * sx), int(y2 * sy))
            color = _color_for_index(i)
            cv2.rectangle(board, p1, p2, color, 2)
            cls = "obj"
            conf = 0.0
            if isinstance(class_names, np.ndarray) and i < class_names.shape[0]:
                cls = str(class_names[i])
            if isinstance(confidences, np.ndarray) and i < confidences.shape[0]:
                conf = float(confidences[i])
            _draw_text(board, f"{cls} {conf:.2f}", p1[0], max(20, p1[1] - 6), scale=0.52, color=color, thickness=2)

    return board


def _render_depth_map(artifact: dict[str, Any], width: int, height: int) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    _draw_text(panel, "Depth Map", 20, 32, scale=0.8, color=(255, 255, 255), thickness=2)

    depth_map = artifact.get("depth_map")
    if not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
        _draw_text(panel, "No depth map available for this frame.", 20, height // 2, scale=0.7, color=(180, 180, 180))
        return panel

    depth = depth_map.astype(np.float32)
    finite = np.isfinite(depth)
    if not finite.any():
        _draw_text(panel, "Depth map values are non-finite.", 20, height // 2, scale=0.7, color=(180, 180, 180))
        return panel

    d_min = float(np.percentile(depth[finite], 2.0))
    d_max = float(np.percentile(depth[finite], 98.0))
    if d_max <= d_min:
        d_max = d_min + 1e-6

    norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
    depth_u8 = (norm * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    color = cv2.resize(color, (width, height), interpolation=cv2.INTER_LINEAR)

    panel = color
    _draw_text(panel, "Depth Map", 20, 32, scale=0.8, color=(255, 255, 255), thickness=2)
    _draw_text(panel, f"p02={d_min:.3f}  p98={d_max:.3f}", 20, 58, scale=0.55, color=(255, 255, 255))
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Offline replay from jetson + edge JSONL logs. Optionally consumes detached "
            "visualization artifacts to render YOLO mask board and depth-map videos."
        )
    )
    parser.add_argument("--jetson-input", required=True)
    parser.add_argument("--edge-input", required=True)
    parser.add_argument("--artifact-dir", default="logs/visualization_artifacts")
    parser.add_argument("--telemetry-output", default="logs/session_replay.mp4")
    parser.add_argument("--yolo-output", default="logs/session_yolo_masks.mp4")
    parser.add_argument("--depth-output", default="logs/session_depth_map.mp4")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--show-window",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--render-telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    jetson_rows = _load_jsonl_rows(args.jetson_input)
    edge_rows = _load_jsonl_rows(args.edge_input)
    frames = _build_frames(jetson_rows, edge_rows)
    if not frames:
        raise SystemExit("No correlated replay frames found.")

    total_frames = len(frames) if args.max_frames <= 0 else min(len(frames), int(args.max_frames))
    width = max(640, int(args.width))
    height = max(480, int(args.height))
    fps = max(1, int(args.fps))

    Path(args.telemetry_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.yolo_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.depth_output).parent.mkdir(parents=True, exist_ok=True)

    telemetry_writer = None
    if args.render_telemetry:
        telemetry_writer = cv2.VideoWriter(
            args.telemetry_output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (width, height),
        )
        if not telemetry_writer.isOpened():
            raise SystemExit(f"Failed to open telemetry video writer: {args.telemetry_output}")

    yolo_writer = cv2.VideoWriter(
        args.yolo_output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not yolo_writer.isOpened():
        raise SystemExit(f"Failed to open YOLO video writer: {args.yolo_output}")

    depth_writer = cv2.VideoWriter(
        args.depth_output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not depth_writer.isOpened():
        raise SystemExit(f"Failed to open depth video writer: {args.depth_output}")

    artifact_dir = Path(args.artifact_dir)
    artifact_hits = 0

    for i in range(total_frames):
        rec = frames[i]
        artifact = _load_artifact(artifact_dir, rec.trace_id)

        if artifact is not None:
            artifact_hits += 1
            yolo_frame = _render_yolo_board(artifact, width, height)
            depth_frame = _render_depth_map(artifact, width, height)
        else:
            yolo_frame = np.zeros((height, width, 3), dtype=np.uint8)
            depth_frame = np.zeros((height, width, 3), dtype=np.uint8)
            _draw_text(yolo_frame, "YOLO Segmentation Masks (Blackboard)", 20, 32, scale=0.8, color=(255, 255, 255), thickness=2)
            _draw_text(yolo_frame, f"No artifact for trace_id={rec.trace_id[:12]}...", 20, 80, scale=0.62, color=(180, 180, 180))
            _draw_text(depth_frame, "Depth Map", 20, 32, scale=0.8, color=(255, 255, 255), thickness=2)
            _draw_text(depth_frame, f"No artifact for trace_id={rec.trace_id[:12]}...", 20, 80, scale=0.62, color=(180, 180, 180))

        yolo_writer.write(yolo_frame)
        depth_writer.write(depth_frame)

        if telemetry_writer is not None:
            tele = np.zeros((height, width, 3), dtype=np.uint8)
            _draw_text(tele, "Offline Telemetry Replay", 20, 34, scale=0.82, color=(255, 255, 255), thickness=2)
            _draw_text(tele, "Detached mode: reads JSONL and artifact files after the run.", 20, 62, scale=0.52, color=(160, 160, 160))
            stale_color = (0, 220, 255) if rec.is_stale else (60, 180, 60)
            cv2.rectangle(tele, (width - 220, 18), (width - 24, 56), stale_color, -1)
            _draw_text(tele, "STALE" if rec.is_stale else "FRESH", width - 170, 45, scale=0.66, color=(0, 0, 0), thickness=2)
            _draw_text(tele, f"frame {i + 1}/{total_frames}", 20, 110)
            _draw_text(tele, f"trace_id: {rec.trace_id}", 20, 138, scale=0.5)
            _draw_text(tele, f"age_ms: {rec.age_ms:.2f}  threshold_ms: {rec.stale_threshold_ms:.0f}", 20, 176, scale=0.68, color=(200, 230, 255), thickness=2)
            _draw_text(tele, f"yolo_inference_ms: {rec.yolo_inference_ms:.2f}", 20, 214, color=(80, 255, 255))
            _draw_text(tele, f"depth_inference_ms: {rec.depth_inference_ms:.2f}", 20, 242, color=(255, 180, 80))
            _draw_text(tele, f"capture_to_edge_ms: {rec.capture_to_edge_ms:.2f}", 20, 270)
            _draw_text(tele, f"edge_to_done_ms: {rec.edge_to_done_ms:.2f}", 20, 298)
            _draw_text(tele, f"yolo_object_count: {rec.object_count}", 20, 326)
            _draw_text(tele, f"artifact_available: {'yes' if artifact is not None else 'no'}", 20, 354)
            telemetry_writer.write(tele)

        if args.show_window:
            cv2.imshow("yolo_masks_blackboard", yolo_frame)
            cv2.imshow("depth_map", depth_frame)
            if telemetry_writer is not None:
                cv2.imshow("telemetry", tele)
            key = cv2.waitKey(max(1, int(1000 / fps)))
            if key == ord("q") or key == 27:
                break

    yolo_writer.release()
    depth_writer.release()
    if telemetry_writer is not None:
        telemetry_writer.release()
    if args.show_window:
        cv2.destroyAllWindows()

    print("Replay rendered successfully")
    print(f"- frames={total_frames}")
    print(f"- artifact_hits={artifact_hits}")
    print(f"- yolo_output={args.yolo_output}")
    print(f"- depth_output={args.depth_output}")
    if telemetry_writer is not None:
        print(f"- telemetry_output={args.telemetry_output}")
    print("- mode=offline_detached")


if __name__ == "__main__":
    main()

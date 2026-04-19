from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import shutil

import av
import numpy as np


@dataclass
class VisualizationArtifact:
    trace_id: str
    inference_ts_ms: int
    frame_bgr: np.ndarray
    yolo_masks: np.ndarray
    class_ids: np.ndarray
    class_names: np.ndarray
    confidences: np.ndarray
    bboxes: np.ndarray
    depth_map: np.ndarray | None


class AsyncVisualizationDumpWriter:
    def __init__(self, output_dir: str, max_queue_size: int = 8, clear_existing: bool = True) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        if clear_existing:
            self._clear_output_dir()
        self._queue: queue.Queue[VisualizationArtifact | None] = queue.Queue(maxsize=max_queue_size)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _clear_output_dir(self) -> None:
        for item in self._output_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    def submit(
        self,
        *,
        trace_id: str,
        inference_ts_ms: int,
        frame_bgr: np.ndarray,
        yolo_objects: list[dict[str, Any]],
        depth_map: np.ndarray | None,
    ) -> None:
        frame_copy = np.ascontiguousarray(frame_bgr.copy())
        frame_h, frame_w = frame_copy.shape[:2]

        masks: list[np.ndarray] = []
        class_ids: list[int] = []
        class_names: list[str] = []
        confidences: list[float] = []
        bboxes: list[list[float]] = []

        for obj in yolo_objects:
            mask = obj.get("mask")
            if not isinstance(mask, np.ndarray):
                continue

            mask_bool = mask.astype(bool)
            if mask_bool.shape != (frame_h, frame_w):
                vf = av.VideoFrame.from_ndarray(mask_bool.astype(np.uint8) * 255, format="gray")
                vf = vf.reformat(width=frame_w, height=frame_h, format="gray")
                resized = vf.to_ndarray(format="gray")
                mask_bool = resized > 0

            masks.append(mask_bool.astype(np.uint8))
            class_ids.append(int(obj.get("class_id", -1)))
            class_names.append(str(obj.get("class_name", "")))
            confidences.append(float(obj.get("confidence", 0.0)))
            bboxes.append([float(v) for v in obj.get("bbox_xyxy", [0, 0, 0, 0])])

        if masks:
            yolo_mask_stack = np.stack(masks, axis=0)
        else:
            yolo_mask_stack = np.zeros((0, frame_h, frame_w), dtype=np.uint8)

        if depth_map is not None and isinstance(depth_map, np.ndarray):
            depth_copy = depth_map.astype(np.float16)
        else:
            depth_copy = None

        artifact = VisualizationArtifact(
            trace_id=trace_id,
            inference_ts_ms=int(inference_ts_ms),
            frame_bgr=frame_copy,
            yolo_masks=yolo_mask_stack,
            class_ids=np.asarray(class_ids, dtype=np.int32),
            class_names=np.asarray(class_names, dtype=object),
            confidences=np.asarray(confidences, dtype=np.float32),
            bboxes=np.asarray(bboxes, dtype=np.float32),
            depth_map=depth_copy,
        )

        try:
            self._queue.put_nowait(artifact)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(artifact)
            except queue.Full:
                return

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            try:
                self._write_item(item)
            finally:
                self._queue.task_done()

    def _write_item(self, item: VisualizationArtifact) -> None:
        file_path = self._output_dir / f"{item.trace_id}.npz"
        np.savez_compressed(
            file_path,
            trace_id=item.trace_id,
            inference_ts_ms=item.inference_ts_ms,
            frame_bgr=item.frame_bgr,
            yolo_masks=item.yolo_masks,
            class_ids=item.class_ids,
            class_names=item.class_names,
            confidences=item.confidences,
            bboxes=item.bboxes,
            depth_map=item.depth_map,
        )

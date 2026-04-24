from __future__ import annotations

import numpy as np
import pytest

from services.edge_gateway.triton_infer import (
    compute_object_depth_overlap,
    decode_depth_output,
    decode_yolo_segmentation,
)


def test_decode_depth_output_returns_map_and_percentiles() -> None:
    result = {
        "status": "ok",
        "raw_outputs": {
            "depth": np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
        },
    }

    decoded = decode_depth_output(result, output_name="depth")

    assert decoded["status"] == "ok"
    assert decoded["output_shape"] == [1, 1, 2, 2]
    assert decoded["depth_map"].shape == (2, 2)
    assert decoded["depth_percentiles"]["p50"] == 2.5


def test_decode_yolo_segmentation_extracts_objects() -> None:
    result = {
        "status": "ok",
        "raw_outputs": {
            "output0": np.array([[[0.0, 0.0, 4.0, 4.0, 0.9, 2.0, 10.0]]], dtype=np.float32),
            "output1": np.ones((1, 1, 2, 2), dtype=np.float32),
        },
    }

    decoded = decode_yolo_segmentation(
        result,
        input_width=4,
        input_height=4,
        score_threshold=0.25,
        mask_threshold=0.5,
        max_objects=10,
    )

    assert decoded["status"] == "ok"
    assert decoded["object_count"] == 1
    obj = decoded["objects"][0]
    assert obj["class_id"] == 2
    assert obj["class_name"] == "car"
    assert obj["confidence"] == pytest.approx(0.9)
    assert obj["mask_pixels_input"] > 0


def test_compute_object_depth_overlap_uses_median_and_spread() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    yolo_decoded = {
        "status": "ok",
        "object_count": 1,
        "objects": [
            {
                "class_id": 1,
                "class_name": "bicycle",
                "confidence": 0.95,
                "bbox_xyxy": [1.0, 1.0, 3.0, 3.0],
                "mask": mask,
            }
        ],
    }

    depth_map = np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 10.0, 20.0, 1.0],
            [1.0, 30.0, 40.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    depth_decoded = {
        "status": "ok",
        "depth_map": depth_map,
        "depth_percentiles": {
            "p10": 0.0,
            "p90": 100.0,
        },
    }

    overlap = compute_object_depth_overlap(
        yolo_decoded=yolo_decoded,
        depth_decoded=depth_decoded,
        frame_height=4,
        frame_width=4,
        yolo_input_width=4,
        yolo_input_height=4,
        near_threshold=0.35,
        far_threshold=0.65,
    )

    assert overlap["status"] == "ok"
    assert overlap["object_count"] == 1

    obj = overlap["objects"][0]
    assert obj["class_name"] == "bicycle"
    assert obj["depth_median"] == 25.0
    assert obj["depth_p10"] == 13.0
    assert obj["depth_p90"] == 37.0
    assert obj["depth_spread_p90_p10"] == 24.0
    assert obj["depth_band"] == "near"

from __future__ import annotations

from services.edge_gateway.metadata import build_metadata


def test_metadata_stale_flag() -> None:
    timings_ms = {
        "capture_to_edge_rx_ms": 20,
        "edge_rx_to_inference_done_ms": 140,
        "age_ms_reconstructed": 160,
        "yolo": {"inference_ms": 18.5},
        "depth": {"inference_ms": 27.2},
    }
    md = build_metadata(
        trace_id="x",
        capture_ts_ms=100,
        edge_rx_ts_ms=120,
        inference_ts_ms=260,
        detections={},
        stale_threshold_ms=120,
        timings_ms=timings_ms,
    )
    assert md["age_ms"] == 160
    assert md["is_stale"] is True
    assert md["timings_ms"]["age_ms_reconstructed"] == md["age_ms"]

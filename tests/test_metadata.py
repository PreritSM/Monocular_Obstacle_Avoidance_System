from __future__ import annotations

from services.edge_gateway.metadata import build_metadata


def test_metadata_stale_flag() -> None:
    md = build_metadata(
        trace_id="x",
        capture_ts_ms=100,
        edge_rx_ts_ms=120,
        inference_ts_ms=260,
        detections={},
        stale_threshold_ms=120,
    )
    assert md["age_ms"] == 160
    assert md["is_stale"] is True

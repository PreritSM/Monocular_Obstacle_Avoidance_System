from __future__ import annotations


def build_metadata(
    *,
    trace_id: str,
    capture_ts_ms: int,
    edge_rx_ts_ms: int,
    inference_ts_ms: int,
    detections: dict,
    stale_threshold_ms: int,
) -> dict:
    age_ms = max(0, inference_ts_ms - capture_ts_ms)
    return {
        "trace_id": trace_id,
        "capture_ts_ms": capture_ts_ms,
        "edge_rx_ts_ms": edge_rx_ts_ms,
        "inference_ts_ms": inference_ts_ms,
        "age_ms": age_ms,
        "is_stale": age_ms > stale_threshold_ms,
        "stale_threshold_ms": stale_threshold_ms,
        "detections": detections,
    }

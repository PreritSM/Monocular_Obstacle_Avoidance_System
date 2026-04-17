from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
from typing import Any

import orjson
import yaml


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    idx = int((len(values) - 1) * p)
    return sorted(values)[idx]


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("rb") as f:
        for line in f:
            rows.append(orjson.loads(line))
    return rows


def extract_latency_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latency_rows: list[dict[str, Any]] = []

    for row in rows:
        event = row.get("event")
        if event == "metadata_rx":
            message = row.get("message")
            if not isinstance(message, str):
                continue
            try:
                payload = orjson.loads(message)
            except orjson.JSONDecodeError:
                continue
            if "age_ms" in payload:
                latency_rows.append(payload)
        elif event == "inference_done":
            latency_rows.append(row)

    return latency_rows


def _nested_number(row: dict[str, Any], path: list[str]) -> float | None:
    cur: Any = row
    for part in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--thresholds", required=True)
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    with Path(args.thresholds).open("r", encoding="utf-8") as f:
        th = yaml.safe_load(f)

    inf = extract_latency_rows(rows)
    age = [float(r.get("age_ms", 0)) for r in inf]
    stale = [r for r in inf if bool(r.get("is_stale"))]
    yolo_inf = [
        v
        for v in (_nested_number(r, ["timings_ms", "yolo", "inference_ms"]) for r in inf)
        if v is not None
    ]
    depth_inf = [
        v
        for v in (_nested_number(r, ["timings_ms", "depth", "inference_ms"]) for r in inf)
        if v is not None
    ]
    age_reconstructed = [
        v
        for v in (_nested_number(r, ["timings_ms", "age_ms_reconstructed"]) for r in inf)
        if v is not None
    ]

    med = median(age) if age else 0.0
    p95 = percentile(age, 0.95)
    p99 = percentile(age, 0.99)
    yolo_med = median(yolo_inf) if yolo_inf else 0.0
    depth_med = median(depth_inf) if depth_inf else 0.0

    stale_rate = (len(stale) / len(inf) * 100.0) if inf else 0.0
    age_reconstructed_med = median(age_reconstructed) if age_reconstructed else 0.0
    age_vs_reconstructed_residual = med - age_reconstructed_med

    print("Latency summary")
    print(f"- median_ms={med:.2f} (target<{th['latency']['median_ms_max']})")
    print(f"- p95_ms={p95:.2f} (target<{th['latency']['p95_ms_max']})")
    print(f"- p99_ms={p99:.2f} (target<{th['latency']['p99_ms_max']})")
    print(f"- stale_rate_pct={stale_rate:.2f} (target<{th['stale']['stale_rate_nominal_max_pct']})")
    print(f"- yolo_inference_median_ms={yolo_med:.2f}")
    print(f"- depth_inference_median_ms={depth_med:.2f}")
    if age_reconstructed:
        print(f"- age_reconstructed_median_ms={age_reconstructed_med:.2f}")
        print(f"- age_minus_reconstructed_median_ms={age_vs_reconstructed_residual:.2f}")


if __name__ == "__main__":
    main()

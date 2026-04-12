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

    med = median(age) if age else 0.0
    p95 = percentile(age, 0.95)
    p99 = percentile(age, 0.99)

    stale_rate = (len(stale) / len(inf) * 100.0) if inf else 0.0

    print("Latency summary")
    print(f"- median_ms={med:.2f} (target<{th['latency']['median_ms_max']})")
    print(f"- p95_ms={p95:.2f} (target<{th['latency']['p95_ms_max']})")
    print(f"- p99_ms={p99:.2f} (target<{th['latency']['p99_ms_max']})")
    print(f"- stale_rate_pct={stale_rate:.2f} (target<{th['stale']['stale_rate_nominal_max_pct']})")


if __name__ == "__main__":
    main()

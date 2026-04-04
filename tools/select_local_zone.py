from __future__ import annotations

import argparse
import math

import boto3

# Approximate coords for ranking by geographic closeness when direct RTT probes are not available.
CITY_COORDS = {
    "Henrietta, NY": (43.0387, -77.6127),
    "New York, NY": (40.7128, -74.0060),
    "Boston, MA": (42.3601, -71.0589),
    "Philadelphia, PA": (39.9526, -75.1652),
}

# Heuristic coordinates for common us-east-1 local zone metro tags.
LZ_COORDS = {
    "nyc": (40.7128, -74.0060),
    "bos": (42.3601, -71.0589),
    "phl": (39.9526, -75.1652),
    "atl": (33.7490, -84.3880),
    "mia": (25.7617, -80.1918),
}


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    r = 6371
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(x))


def infer_zone_coords(zone_name: str) -> tuple[float, float] | None:
    parts = zone_name.split("-")
    if len(parts) < 4:
        return None
    metro = parts[3]
    return LZ_COORDS.get(metro)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="g6.xlarge")
    parser.add_argument("--city", default="Henrietta, NY")
    args = parser.parse_args()

    ec2 = boto3.client("ec2", region_name=args.region)

    azs = ec2.describe_availability_zones(AllAvailabilityZones=True)["AvailabilityZones"]
    local_zones = [z for z in azs if z.get("ZoneType") == "local-zone" and z["RegionName"] == args.region]

    offerings = ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[
            {"Name": "instance-type", "Values": [args.instance_type]},
            {"Name": "location", "Values": [z["ZoneName"] for z in local_zones]},
        ],
    )["InstanceTypeOfferings"]
    allowed = {x["Location"] for x in offerings}

    base = CITY_COORDS.get(args.city)
    rows: list[tuple[str, float | None]] = []
    for zone in local_zones:
        name = zone["ZoneName"]
        if name not in allowed:
            continue
        if base is None:
            rows.append((name, None))
            continue
        coords = infer_zone_coords(name)
        rows.append((name, haversine_km(base, coords) if coords else None))

    rows.sort(key=lambda x: 10_000 if x[1] is None else x[1])

    print(f"Candidate Local Zones in {args.region} for {args.instance_type}:")
    for name, distance in rows:
        suffix = "distance_unknown" if distance is None else f"approx_distance_km={distance:.1f}"
        print(f"- {name} ({suffix})")


if __name__ == "__main__":
    main()

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

FALLBACK_GPU_TYPES = [
    "g5.xlarge",
    "g5.2xlarge",
    "g4dn.xlarge",
    "g4dn.2xlarge",
]


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


def list_instance_offerings(ec2, locations: list[str], instance_type: str) -> set[str]:
    if not locations:
        return set()
    offerings = ec2.describe_instance_type_offerings(
        LocationType="availability-zone",
        Filters=[
            {"Name": "instance-type", "Values": [instance_type]},
            {"Name": "location", "Values": locations},
        ],
    )["InstanceTypeOfferings"]
    return {x["Location"] for x in offerings}


def zone_distance_km(zone_name: str, city_coords: tuple[float, float] | None) -> float | None:
    if city_coords is None:
        return None
    coords = infer_zone_coords(zone_name)
    if coords is None:
        return None
    return haversine_km(city_coords, coords)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default="g6.xlarge")
    parser.add_argument("--city", default="Henrietta, NY")
    parser.add_argument(
        "--fallback-instance-types",
        default=",".join(FALLBACK_GPU_TYPES),
        help="Comma-separated fallback instance types to probe when requested type is unavailable.",
    )
    args = parser.parse_args()

    ec2 = boto3.client("ec2", region_name=args.region)

    azs = ec2.describe_availability_zones(AllAvailabilityZones=True)["AvailabilityZones"]
    local_zones = [z for z in azs if z.get("ZoneType") == "local-zone" and z["RegionName"] == args.region]
    zone_names = [z["ZoneName"] for z in local_zones]
    opted_in_zone_names = [z["ZoneName"] for z in local_zones if z.get("OptInStatus") == "opted-in"]
    fallback_types = [x.strip() for x in args.fallback_instance_types.split(",") if x.strip()]

    if not local_zones:
        print(f"No Local Zones discovered in region {args.region} for this account.")
        return

    if not opted_in_zone_names:
        print(f"No Local Zones are opted-in for this account in {args.region}.")
        print("Visible Local Zones and opt-in status:")
        for zone in sorted(local_zones, key=lambda z: z["ZoneName"]):
            print(f"- {zone['ZoneName']}: {zone.get('OptInStatus', 'unknown')}")
        return

    allowed = list_instance_offerings(ec2, opted_in_zone_names, args.instance_type)

    base = CITY_COORDS.get(args.city)
    rows: list[tuple[str, float | None]] = []
    for zone in local_zones:
        name = zone["ZoneName"]
        if name not in allowed:
            continue
        rows.append((name, zone_distance_km(name, base)))

    rows.sort(key=lambda x: 10_000 if x[1] is None else x[1])

    print(f"Candidate Local Zones in {args.region} for {args.instance_type}:")
    if not rows:
        print("- none")
        print("")
        print(
            f"Requested instance type {args.instance_type} is not offered in any visible Local Zone in {args.region}."
        )
        if fallback_types:
            print("Fallback probe results:")
            for instance_type in fallback_types:
                offered = list_instance_offerings(ec2, opted_in_zone_names, instance_type)
                if not offered:
                    print(f"- {instance_type}: none")
                    continue
                ranked = sorted(
                    [(zone, zone_distance_km(zone, base)) for zone in offered],
                    key=lambda x: 10_000 if x[1] is None else x[1],
                )
                preview = ", ".join(
                    f"{zone} ({'distance_unknown' if dist is None else f'{dist:.1f} km'})"
                    for zone, dist in ranked[:3]
                )
                print(f"- {instance_type}: {preview}")
        return

    for name, distance in rows:
        suffix = "distance_unknown" if distance is None else f"approx_distance_km={distance:.1f}"
        print(f"- {name} ({suffix})")


if __name__ == "__main__":
    main()

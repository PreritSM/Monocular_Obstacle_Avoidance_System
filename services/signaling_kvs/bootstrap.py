from __future__ import annotations

import argparse

import boto3


def ensure_channel(channel_name: str, region: str) -> str:
    kv = boto3.client("kinesisvideo", region_name=region)
    try:
        arn = kv.describe_signaling_channel(ChannelName=channel_name)["ChannelInfo"]["ChannelARN"]
    except kv.exceptions.ResourceNotFoundException:
        arn = kv.create_signaling_channel(ChannelName=channel_name)["ChannelARN"]
    return arn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel-name", required=True)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    channel_arn = ensure_channel(args.channel_name, args.region)

    kv = boto3.client("kinesisvideo", region_name=args.region)
    endpoint_cfg = kv.get_signaling_channel_endpoint(
        ChannelARN=channel_arn,
        SingleMasterChannelEndpointConfiguration={
            "Protocols": ["WSS", "HTTPS"],
            "Role": "MASTER",
        },
    )

    by_proto = {x["Protocol"]: x["ResourceEndpoint"] for x in endpoint_cfg["ResourceEndpointList"]}

    signaling = boto3.client("kinesis-video-signaling", region_name=args.region, endpoint_url=by_proto["HTTPS"])
    ice = signaling.get_ice_server_config(ChannelARN=channel_arn)

    print(f"channel_arn={channel_arn}")
    print(f"wss_endpoint={by_proto.get('WSS')}")
    print(f"https_endpoint={by_proto.get('HTTPS')}")
    print("ice_servers=")
    for item in ice.get("IceServerList", []):
        print(f"- uris={item.get('Uris', [])}")


if __name__ == "__main__":
    main()

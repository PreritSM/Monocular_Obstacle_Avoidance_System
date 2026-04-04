from __future__ import annotations


class AwsKvsSignalingClient:
    def __init__(self, channel_name: str, region: str, client_id: str) -> None:
        self.channel_name = channel_name
        self.region = region
        self.client_id = client_id

    async def connect(self) -> None:
        raise NotImplementedError(
            "AWS KVS signaling transport is account-specific. Use self_hosted mode for now."
        )

# AWS Kinesis Video Streams signaling path (Phase 1 scaffold)

This folder keeps the KVS signaling path explicit in the repository so you can switch from self-hosted signaling without changing the rest of the app contracts.

Phase 1 includes:

- shared signaling interface in sender/gateway apps
- dedicated `aws_kvs` config files
- placeholder client class with clear extension point

To complete KVS signaling in your account, implement:

1. Channel discovery/creation
2. WSS signaling endpoint retrieval
3. ICE server config retrieval from KVS
4. SDP/ICE message transport over the KVS signaling channel

The rest of the media and inference stack remains unchanged.

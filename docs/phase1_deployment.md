# Phase 1 Deployment Blueprint

## 1) Pick Local Zone in us-east-1

Run:

```bash
pip install -r tools/requirements.txt
python -m tools.select_local_zone --region us-east-1 --instance-type g6.xlarge --city "Henrietta, NY"
```

Pick the best candidate with G6 support.

## 2) Provision AWS infra (single EC2 G6, Ubuntu 22.04)

```bash
cd deploy/terraform
cp terraform.tfvars.example terraform.tfvars
# edit availability_zone and ssh_cidr
terraform init
terraform apply
```

## 3) Sync repository to EC2 and run docker stack

```bash
cd deploy/docker
docker compose up -d --build
```

## 4) Start Jetson sender

```bash
python -m jetson_client.app --config configs/jetson.self_hosted.yaml
```

## 5) Evaluate metrics

```bash
python -m tools.analyze_metrics --input logs/edge_session.jsonl --thresholds configs/acceptance.thresholds.yaml
```

## Notes

- Self-hosted signaling is complete for Phase 1 testing.
- AWS KVS signaling is scaffolded and bootstrapped in `services/signaling_kvs/bootstrap.py`.
- Safety/failsafe behavior is intentionally deferred to Phase 2.

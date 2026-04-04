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

## GitHub Actions deployment (all values centralized)

Use workflow `.github/workflows/deploy-phase1.yml` to deploy with all Phase 1 values provided as workflow inputs.

### Required repository secrets

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `EC2_SSH_PRIVATE_KEY` only if you enable `deploy_to_ec2=true`

### Workflow inputs

- `region` (default `us-east-1`)
- `availability_zone` (required, Local Zone AZ like `us-east-1-nyc-1a`)
- `instance_type` (default `g6.xlarge`)
- `ssh_cidr` (default `0.0.0.0/0`)
- `name_prefix` (default `depth-yolo-phase1`)
- `key_name` (optional)
- `apply_changes` (`true`/`false`)
- `deploy_to_ec2` (`true`/`false`)

### What the workflow does

1. Generates `deploy/terraform/terraform.auto.tfvars` from the provided inputs.
2. Runs `terraform init`, `terraform validate`, and `terraform plan`.
3. Applies changes if `apply_changes=true`.
4. Optionally copies the repo and starts Docker Compose on EC2 if `deploy_to_ec2=true`.
5. Publishes deployment output values in the workflow summary.

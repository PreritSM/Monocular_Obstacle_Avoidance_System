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
- `availability_zone` (optional manual override)
- `instance_type` (optional manual override)
- `preferred_instance_types` (priority list for auto-selection)
- `city` (used to rank nearest zone when multiple options are available)
- `ssh_cidr` (default `0.0.0.0/0`)
- `name_prefix` (default `depth-yolo-phase1`)
- `key_name` (optional)
- `terraform_action` (`apply`/`destroy`)
- `deploy_to_ec2` (`true`/`false`)

### What the workflow does

1. Generates `deploy/terraform/terraform.auto.tfvars` from the provided inputs.
2. Runs preflight checks for:
	- discovers all Local Zones in region
	- filters to `opted-in` zones
	- checks offered instance types across opted-in zones
	- auto-selects best zone+instance based on `preferred_instance_types` priority and nearest distance to `city`
	- if both `availability_zone` and `instance_type` are provided, validates and uses them as manual override
3. Runs `terraform init` and `terraform validate`.
4. Runs `terraform plan` as part of the `apply` path.
5. Runs `terraform apply` when `terraform_action=apply`.
6. Runs `terraform destroy` when `terraform_action=destroy`.
7. Optionally copies the repo and starts Docker Compose on EC2 when `terraform_action=apply` and `deploy_to_ec2=true`.
8. Publishes deployment output values in the workflow summary.

If preflight fails with "not opted-in", enable the Local Zone in EC2 Console:

- `EC2` -> `Account attributes` -> `Zones` -> opt in the target Local Zone.

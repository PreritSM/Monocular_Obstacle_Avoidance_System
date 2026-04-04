# Deployment Action Inputs Reference

This document describes every input field for the GitHub Actions workflow `.github/workflows/deploy-phase1.yml`.

## Input fields

### `region`
- Purpose: AWS region where Terraform and AWS API calls run.
- Type: string
- Default: `us-east-1`
- Example: `us-east-1`
- Notes: Keep this aligned with your Local Zone parent region.

### `availability_zone`
- Purpose: Manual zone override for deployment/destroy.
- Type: string
- Default: empty
- Example: `us-east-1-nyc-2a`
- Notes:
  - Optional for `terraform_action=apply` (preflight can auto-select).
  - Useful for `terraform_action=destroy` if state lookup is unavailable.

### `instance_type`
- Purpose: Manual EC2 instance type override.
- Type: string
- Default: empty
- Example: `g6.xlarge`
- Notes:
  - Optional for `terraform_action=apply` (preflight can auto-select from preferred list).
  - If manual `availability_zone` and `instance_type` are both provided, preflight validates and uses them.

### `preferred_instance_types`
- Purpose: Priority list used by preflight auto-selection.
- Type: comma-separated string
- Default: `g6.xlarge,g5.xlarge,g5.2xlarge,g4dn.xlarge,g4dn.2xlarge`
- Example: `g6.xlarge,g5.xlarge,g4dn.xlarge`
- Notes: Order matters; earlier entries are preferred over later ones.

### `city`
- Purpose: City hint used to rank candidate Local Zones by approximate distance.
- Type: string
- Default: `Henrietta, NY`
- Example: `Henrietta, NY`
- Notes: If city is unknown to the heuristic map, distance ranking is reduced and selection falls back mostly to instance priority.

### `ssh_cidr`
- Purpose: CIDR block allowed to SSH into EC2 security group.
- Type: string
- Default: `0.0.0.0/0`
- Example: `203.0.113.8/32`
- Notes: Prefer your public IP `/32` for security.

### `name_prefix`
- Purpose: Prefix for Terraform-created resource names.
- Type: string
- Default: `depth-yolo-phase1`
- Example: `depth-yolo-phase1`
- Notes: Also influences the default backend state key when not explicitly set.

### `key_name`
- Purpose: Optional EC2 key pair name for SSH key-based access.
- Type: string
- Default: empty
- Example: `my-ec2-keypair`
- Notes: Leave empty if you plan to use SSM only.

### `use_remote_backend`
- Purpose: Controls Terraform backend mode.
- Type: choice (`true` or `false`)
- Default: `true`
- Example: `true`
- Notes:
  - `true`: uses S3 backend with DynamoDB locking (recommended).
  - `false`: uses local backend in the runner workspace (ephemeral; not ideal across runs).

### `backend_state_bucket`
- Purpose: S3 bucket name for Terraform remote state.
- Type: string
- Default: empty
- Example: `depth-yolo-terraform-state-prod`
- Notes:
  - Required when `use_remote_backend=true` unless provided through secret `TF_STATE_BUCKET`.
  - Must be a real bucket name, not boolean values.

### `backend_lock_table`
- Purpose: DynamoDB table name for Terraform state locking.
- Type: string
- Default: empty
- Example: `depth-yolo-terraform-locks`
- Notes:
  - Required when `use_remote_backend=true` unless provided through secret `TF_LOCK_TABLE`.
  - Must use partition key `LockID` (String).

### `backend_key`
- Purpose: State object key inside the S3 bucket.
- Type: string
- Default: empty
- Example: `depth-yolo-phase1/us-east-1/terraform.tfstate`
- Notes:
  - Optional.
  - If empty, workflow defaults to: `<name_prefix>/<region>/terraform.tfstate`.
  - Can also be provided via secret `TF_STATE_KEY`.

### `terraform_action`
- Purpose: Selects Terraform execution path.
- Type: choice (`apply` or `destroy`)
- Default: `apply`
- Example: `apply`
- Notes:
  - `apply`: runs preflight selection, init/validate/plan/apply.
  - `destroy`: runs init/validate/destroy and attempts to recover zone from state if needed.

### `deploy_to_ec2`
- Purpose: Controls post-apply remote deployment steps.
- Type: choice (`true` or `false`)
- Default: `true`
- Example: `true`
- Notes:
  - Only effective when `terraform_action=apply`.
  - If `true`, workflow syncs repository to EC2 and starts Docker Compose.
  - Requires secret `EC2_SSH_PRIVATE_KEY`.

## Recommended input sets

## Apply (normal deployment)
- `terraform_action`: `apply`
- `use_remote_backend`: `true`
- `backend_state_bucket`: real S3 bucket
- `backend_lock_table`: real DynamoDB table
- `availability_zone`: empty (let preflight auto-select)
- `instance_type`: empty (let preflight auto-select)
- `deploy_to_ec2`: `true`

## Destroy (teardown)
- `terraform_action`: `destroy`
- `use_remote_backend`: `true`
- `backend_state_bucket`: same bucket used for apply
- `backend_lock_table`: same lock table used for apply
- `backend_key`: same key used for apply (or leave empty if same defaults)
- `availability_zone`: optional (provide if state cannot be read)
- `deploy_to_ec2`: `false`

## Related secrets
- Required:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
- Required when `deploy_to_ec2=true`:
  - `EC2_SSH_PRIVATE_KEY`
- Optional backend fallbacks:
  - `TF_STATE_BUCKET`
  - `TF_LOCK_TABLE`
  - `TF_STATE_KEY`

## Backend bootstrap workflow

Use `.github/workflows/bootstrap-terraform-backend.yml` to create or reuse backend resources before running deploy.

Inputs in this bootstrap workflow:
- `region`
- `state_bucket`
- `lock_table`
- `state_key_prefix`
- `enable_versioning`
- `enforce_sse`
- `block_public_access`

After bootstrap finishes, set these repository secrets:
- `TF_STATE_BUCKET`
- `TF_LOCK_TABLE`
- Optional: `TF_STATE_KEY`

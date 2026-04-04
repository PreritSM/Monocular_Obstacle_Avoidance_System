variable "region" {
  type    = string
  default = "us-east-1"
}

variable "availability_zone" {
  type        = string
  description = "Local Zone AZ name (example: us-east-1-nyc-1a)."
}

variable "instance_type" {
  type    = string
  default = "g6.xlarge"
}

variable "ssh_cidr" {
  type    = string
  default = "0.0.0.0/0"
}

variable "name_prefix" {
  type    = string
  default = "depth-yolo-phase1"
}

variable "key_name" {
  type        = string
  description = "Optional EC2 key pair name. Leave empty to use SSM only."
  default     = ""
}

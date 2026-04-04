output "instance_id" {
  value = aws_instance.edge.id
}

output "public_ip" {
  value = aws_instance.edge.public_ip
}

output "availability_zone" {
  value = aws_instance.edge.availability_zone
}

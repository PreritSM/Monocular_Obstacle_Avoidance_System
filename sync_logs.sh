#!/bin/bash

# 1. Define the EST Timestamp (e.g., 2026-04-24_12-55-30)
TIMESTAMP=$(TZ="America/New_York" date "+%Y-%m-%d_%H-%M-%S")

# 2. Define Local Target Directory
TARGET_DIR="./logs/$TIMESTAMP"

# 3. Create the local directory
echo "Creating directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

echo "------------------------------------------------"
echo "Starting transfer from remote server..."

# 4. Command 1: Sync the visualization artifacts folder
# (Trailing slash on source copies contents into the target)
rsync -avz -e "ssh -p 37099" \
  root@184.144.255.168:/root/Monocular_Obstacle_Avoidance_System/logs/visualization_artifacts/ \
  "$TARGET_DIR/visualization_artifacts/"

# 5. Command 2: Sync the edge_session.jsonl file
rsync -avz -e "ssh -p 37099" \
  root@184.144.255.168:/root/Monocular_Obstacle_Avoidance_System/logs/edge_session.jsonl \
  "$TARGET_DIR/"

# 6. Final Status Check
if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "SUCCESS! All data saved to: $TARGET_DIR"
    ls -F "$TARGET_DIR"
    echo "------------------------------------------------"
else
    echo "------------------------------------------------"
    echo "ERROR: One or more transfers failed."
    echo "------------------------------------------------"
    exit 1
fi
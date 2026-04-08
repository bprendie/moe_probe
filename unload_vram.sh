#!/bin/bash

# Target container name
CONTAINER_NAME="granite-moe-demo"

echo -e "\033[1;31m[System]\033[0m Unloading VRAM from $CONTAINER_NAME..."

# Kill all python processes inside the container
# -9 (SIGKILL) ensures the GPU driver immediately recognizes the process is gone
docker exec $CONTAINER_NAME pkill -9 python

echo -e "\033[1;32m[System]\033[0m Python processes terminated."
echo -e "\033[1;34m[System]\033[0m Current GPU Memory Usage:"

# Show the current VRAM usage to confirm success
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print "Used: " $1 "MB / Total: " $2 "MB (" $1/$2*100 "% usage)"}'

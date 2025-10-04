#!/bin/bash
# Kill all training processes

echo "Killing training processes..."
pkill -f "train.py"
sleep 2
echo "Done"

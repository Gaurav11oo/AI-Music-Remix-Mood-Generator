#!/usr/bin/env bash

echo "Installing system dependencies..."

apt-get update
apt-get install -y ffmpeg

echo "System setup complete"

#!/bin/bash
# This script sets up and runs the IAI-IPS Twine Cognition prototype

set -e  # Exit on error

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running Twine Cognition prototype..."
python main.py
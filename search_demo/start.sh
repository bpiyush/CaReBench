#!/bin/bash

# Text-to-Video Search Demo - Quick Start Script

echo "================================================"
echo "Text-to-Video Search Demo Setup"
echo "================================================"
echo ""

# Step 1: Check if we're in the right directory
if [ ! -f "backend.py" ]; then
    echo "Error: backend.py not found. Please run this script from the search_demo directory."
    exit 1
fi

# Step 2: Install dependencies
echo "Step 1: Installing dependencies..."
pip install flask flask-cors faiss-cpu
echo ""

# Step 3: Check setup
echo "Step 2: Checking setup..."
python check_setup.py
echo ""

# Step 4: Start the backend
echo "Step 3: Starting backend server..."
echo ""
echo "The server will start on http://0.0.0.0:5000"
echo "Open http://localhost:5000 in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python backend.py


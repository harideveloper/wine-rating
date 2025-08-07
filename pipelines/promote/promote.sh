#!/bin/bash

# Simple Model Promotion Pipeline
echo "Model Promotion Pipeline"
echo "======================="

# Set environment variables to match constants.py defaults
export SOURCE_PROJECT_ID="dev2-ea8f"
export TARGET_PROJECT_ID="dev1-bfa7"
export REGION="europe-west2"
export MODEL_DISPLAY_NAME="wine-quality-online-prediction-model"
export PROMOTION_THRESHOLD="0.9"
export APPLICATION="wine-quality"
export TYPE="promotion"
export HARNESS_BUILD_ID="local"
export IS_LOCAL="true"
export PIPELINE_SA="1035860259529-compute@developer.gserviceaccount.com"

# Create artifacts directory
mkdir -p artifacts

echo ""
echo "Step 1: Fetch Model"
echo "-------------------"
python fetch.py
if [ $? -ne 0 ]; then
    echo "Fetch failed"
    exit 1
fi

echo ""
echo "Step 2: Check Promotion Gate"
echo "----------------------------"
python promote_gate.py
if [ $? -ne 0 ]; then
    echo "Gate failed - pipeline stopped"
    exit 1
fi

echo ""
echo "Step 3: Register Model"
echo "----------------------"
python register.py
if [ $? -ne 0 ]; then
    echo "Registration failed"
    exit 1
fi

echo ""
echo "Pipeline completed successfully"
echo ""
echo "Generated artifacts:"
ls -la artifacts/

# Show final result
if [ -f "artifacts/registration.json" ]; then
    echo ""
    echo "Final Result:"
    echo "Source: $(cat artifacts/registration.json | python -c "import sys, json; print(json.load(sys.stdin)['source_model_uri'])")"
    echo "Target: $(cat artifacts/registration.json | python -c "import sys, json; print(json.load(sys.stdin)['target_model_uri'])")"
fi
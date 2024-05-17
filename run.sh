#!/bin/bash
chmod +x run.sh

echo "[INFO] Running data processing..."
python3 src/data_processing.py

echo "[INFO] Training the model..."
python3 src/train.py

echo "[INFO] Evaluating the model..."
python3 src/evaluate.py

echo "[INFO] All scripts executed successfully."
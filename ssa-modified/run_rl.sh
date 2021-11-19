#!/bin/bash
cd src
echo "[INFO] RUN RL"
python train.py --display none --explore none --no-qp --mode rl >> logger.txt
#!/bin/bash
cd src
echo "[INFO] RUN RL"
python test_one.py --display none --explore none --no-qp --mode rl >> logger.txt
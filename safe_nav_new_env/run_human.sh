#!/bin/bash
cd src
echo "[INFO] RUN HUMAN"
for i in $(seq .1 .1 .9)
do
    python train.py --display none --explore none --no-qp --mode rl --isHumanBuffer True --replaceRatio $i >> logger.txt
done
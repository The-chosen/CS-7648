#!/bin/bash
cd src
echo "[INFO] RUN SAFE"
for i in $(seq .1 .1 .9)
do
    python test_one.py --display none --explore none --no-qp --mode safe --replaceRatio $i >> logger.txt
done
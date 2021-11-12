#!/bin/bash
cd ~/tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
sha256sum Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
source ~/.bashrc
cd ~/CS-7648/Safe-and-Sample-efficient-Reinforcement-Learning-for-Clustered-Dynamic-Uncertain-Environments/
conda create -n safe-rl python=3.7.9 -y
conda activate safe-rl
pip install -r requirements.txt


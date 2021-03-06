# Safe-and-Sample-efficient-Reinforcement-Learning-for-Clustered-Dynamic-Uncertain-Environments

## Table of Contents
- [Introduction](#Introduction)
- [Install](#install)
- [Usage](#usage)

## Introduction
We provide code for evaluate the safety and sample-efficiency of our proposed RL framework.

For safety, we use Safe Set Algorithm (SSA).   
For efficiency, there are more strategies you can choose:  
1, Adapting SSA;  
2, Exploration (PSN, RND, None);  
3, Learning from SSA;  

The safety and efficiency results of all models are shown below
![safety_result](docs/safety_result.png)
![efficiency_result](docs/efficiency_result.png)

We also provide visual tools.
![visulization](docs/visualization.png)


## Install

```
conda create -n safe-rl
conda install python=3.7.9
pip install tensorflow
pip install keras
pip install matplotlib
pip install gym
pip install cvxopt
```

## Usage

```
python test_one.py --display {none, turtle} --explore {none, psn, rnd} --no-qp --no-ssa-buffer
python test_one.py --display {none, turtle} --explore {none, psn, rnd} --qp --no-ssa-buffer
python test_one.py --display {none, turtle} --explore {none, psn, rnd} --no-qp --ssa-buffer
```
- `--display` can be either `none` or `turtle` (visulization).
- `--explore` specifies the exploration strategy that the robot uses. 
- `--no-qp` means that we use vanilla SSA.
- `--qp` means that we use adapted SSA.
- `--no-ssa-buffer` means that we use the default learning.
- `--ssa-buffer` means that we use the safe learning from SSA demonstrations.



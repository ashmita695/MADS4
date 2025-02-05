
# Multi-Agent Decision S4: Leveraging State Space Models for Offline Multi-Agent Reinforcement Learning

## Overview
Goal-conditioned sequence-based supervised learning with transformers has shown promise in offline reinforcement learning (RL) for single-agent settings. However, extending these methods to offline multi-agent RL (MARL) remains challenging. Existing transformer-based MARL approaches either train agents independently, neglecting multi-agent system dynamics, or rely on centralized transformer models, which face scalability issues. Moreover, transformers inherently struggle with long-term dependencies and computational efficiency. Building on the recent success of Structured State Space Sequence (S4) models, known for their parameter efficiency, faster inference, and superior handling of long context lengths, we propose a novel application of S4-based models to offline MARL tasks. Our method utilizes S4’s efficient convolutional view for offline training and its recurrent dynamics for fast on-policy fine-tuning. To foster scalable cooperation between agents, we sequentially expand the decision-making process, allowing agents to act one after another at each time step. This design promotes bi-directional cooperation, enabling agents to share information via their S4 latent states or memory with minimal communication. Gradients also flow backward through this shared information, linking the current agent’s learning to its predecessor. Experiments on challenging MARL benchmarks, including Multi-Robot Warehouse (RWARE) and StarCraft Multi-Agent Challenge (SMAC), demonstrate that our approach significantly outperforms state-of-the-art offline RL and transformer-based MARL baselines across most tasks.

This codebase is developed based on the work [Decision S4](https://arxiv.org/abs/2306.05167) and the base code from [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://sites.google.com/berkeley.edu/decision-transformer).

Implements the model using variants of the [S4 architecture](https://arxiv.org/abs/2111.00396) from codebase [S4 codebase](https://github.com/state-spaces/s4)


## Instructions

Requires **Python 3.8**, **PyTorch**, and **cudatoolkit**.
Prerequisites are listen in:
```area_prepare/installation.sh```

Set up StarCraft II (2.4.10) and SMAC using the following command.  Alternatively, you could install them manually by following the official link: https://github.com/oxwhirl/smac.

```
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip -P iagreetotheeula SC2.4.10.zip

wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip
mv SMAC_Maps ./StarCraftII/Maps

export SC2PATH=$(pwd)/StarCraftII
pip install git+https://github.com/oxwhirl/smac.git
rm -rf SC2.4.10.zip SMAC_Maps.zip
```
For RWARE environment setup using the gym, follow the instructions available on [RWARE github page](https://github.com/semitable/robotic-warehouse?tab=readme-ov-file#action-space)

## Offline data

The offline SMAC dataset is provided by paper 
“[Offline Pre-trained Multi-Agent Decision Transformer: One Big Sequence Model Tackles All SMAC Tasks](https://arxiv.org/abs/2112.02845v3)". 
For each original large dataset, 1000 episodes are used as our offline dataset which is provided by
([Download link](https://cloud.tsinghua.edu.cn/d/f3c509d7a9d54ccd89c4/)). 

The offline RWARE dataset is provided by the paper [AlberDICE: Addressing OOD Joint Actions in Offline MARL with Alternating DICE](https://arxiv.org/abs/2311.02194)
and is available at [Download link](https://drive.google.com/drive/folders/1e7ttrZzCX2v8HsSMxjhy3Vrd7ZifYSOQ?usp=drive_link)

## How to run

To run an experiment for example:
```
python MADS4/gym/experiment.py --env StarCraft2 --map 2c_vs_64zg --dataset poor --max_iters 50 --num_steps_per_iter 2000
```


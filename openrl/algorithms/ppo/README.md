# Instructions to run ppo_torch.py

## Create conda environment and activate
```
conda create -n ppo python=3.10.12 -y
conda activate ppo
```

## Install packages
```
pip install gym=0.23.0
pip install pygame matplotlib
```

Download Pytorch: https://pytorch.org/get-started/locally/
For Windows:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

## Run
From root, run with command
```
python -m openrl.algorithms.ppo.ppo_torch
```

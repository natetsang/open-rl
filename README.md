# Reinforcement Learning from scratch!
## What's in this repo?
The `open-rl` code repository contains minimalistic implementations of a wide collection of RL algorithms. 
The goal of this repo is to make RL more approachable and easier to learn. As such, code is this repo is optimized for
readability! While simple implementations already exist for Q-networks and vanilla policy gradients, it's difficult
to find easy-to-follow implementations of other algorithms. For many of the algorithms implemented here,
no simple implementations appear to exist whatsoever (at the time of this writing). Interestingly, it's not just
state-of-the-art algorithms that haven't been re-implemented simply. It's also hard to find easy-to-follow 
implementations of foundational algorithms like multi-armed bandits. It's for these reasons why `open-rl` was created!

## Installation
- Make sure you have Python 3.7 or higher installed
- Clone the repo
```
git clone --depth 1 https://github.com/natetsang/open-rl
cd open-rl  # Navigate to root folder
```
- Create a virtual environment (Windows 10)
```
pip install virtualenv # If not already installed
virtualenv venv  # Create virtual environment called 'venv' in the root of the project
venv\Scripts\activate  # Activate environment
```
- Download requirements
```
pip install -r requirements.txt
```
## Algorithms
This repo implements many algorithms, which include the following. 

### Model free learning
#### Policy-based methods
- [x] REINFORCE
- [x] VPG

#### Value-based methods
- [x] DQN
- [x] Double DQN
- [x] Dueling DQN 
- [x] DRQN

#### Actor-critic methods
- [x] A2C
- [x] A3C
- [x] DDPG
- [x] TD3
- [x] SAC
- [x] PPO

### Bandits
#### Multi-armed bandits
- [x] Pure Exploration
- [x] Epsilon Greedy
- [x] Thompson Sampling - Bernoulli
- [x] Thompson Sampling - Gaussian
- [x] UCB

#### Contextual bandits
- [x] Linear UCB
- [ ] Linear Thompson Sampling 
- [x] Neural-network approach

### Model-based methods
- [x] Dyna-Q
- [x] Deep Dyna-Q
- [x] Monte-Carlo Tree Search (MCTS)
- [x] MB + model predictive control
- [x] Model-based Policy Optimization (MBPO)

### Offline (batch) methods
- [x] Conservative Q-learning (CQL)
- [x] MOReL
- [x] Model-based Offline Policy Optimization (MOPO)

### Other
- [x] Behavioral Cloning
- [x] Imitation Learning

## Contributing
`open-rl` is currently still under development and updates  will be made periodically. 
If you're interested in contributing to `open-rl`, please fork the repo and make a pull request. Any support
is much appreciated!

## Citation
If you use this code, please cite it as follows:
```
@misc{Open-RL,
  author = {Nate Tsang},
  title = {Open-RL: An open-source repository of minimalistic implementations of reinforcement learning algorithms},
  year = {2021},
  url = {https://github.com/natetsang/open-rl},
  howpublished = {\url{https://github.com/natetsang/open-rl}},
}
```

## Acknowledgements
This repo would not be possible without the following (tremendous) resources.
* [CS285](http://rail.eecs.berkeley.edu/deeprlcourse/) @ UC Berkeley - taught by Sergey Levine

![Open-RL Logo](https://github.com/natetsang/open-rl/blob/main/logo.PNG)

![GitHub release (latest by date)](https://img.shields.io/github/v/release/natetsang/open-rl?color=blueviolet&style=flat)
![GitHub commits since latest release (by date)](https://img.shields.io/github/commits-since/natetsang/open-rl/latest/main?&style=flat)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?color=lightgrey&style=flat)](http://makeapullrequest.com)

__Open RL__ is code repository that contains minimalistic implementations of a wide collection of reinforcement
learning algorithms. The purpose of this repo is to make RL more approachable and easier to learn. 
As such, code in this repo is optimized for readability and consistency between algorithms. 

Compared to machine learning, RL is still rather niche. As such, finding resources for learning
RL is a bit more difficult. While implementations broadly exist for two algorithms, Q-networks and vanilla policy gradients, 
it's much more difficult to find easy-to-follow implementations of others. 
For many of the algorithms implemented here, no simple implementations appear to exist whatsoever. 
Interestingly, it's not just state-of-the-art algorithms that haven't been re-implemented in an easy-to-follow way. 
It's also hard to find clear implementations of foundational algorithms like multi-armed bandits. 
It's for these reasons why `open-rl` was created! Happy learning!

## Algorithms
In this repo you will find implementations for the following algorithms. 

### Model free learning
#### Policy-based methods
|                           | Discrete              | Continuous                |
| ---                       | :---:                 | :---:                     |
| REINFORCE                 | :heavy_check_mark:    | :heavy_multiplication_x:  |
| REINFORCE w/ baseline     | :heavy_check_mark:    | :heavy_multiplication_x:  |
| VPG                       | :heavy_check_mark:    | :heavy_check_mark:        |

#### Value-based methods
|                           | Discrete              | Continuous                |
| ---                       | :---:                 | :---:                     |
| DQN                       | :heavy_check_mark:    | :heavy_multiplication_x:  |
| Double DQN                | :heavy_check_mark:    | :heavy_multiplication_x:  |
| Dueling DQN               | :heavy_check_mark:    | :heavy_multiplication_x:  |
| DRQN (for POMDPs)         | :heavy_check_mark:    | :heavy_multiplication_x:  |

#### Actor-critic methods
|       | Discrete                  | Continuous                |
| ---   | :---:                     | :---:                     |
| A2C   | :heavy_check_mark:        | :heavy_multiplication_x:  |
| A3C   | :heavy_check_mark:        | :heavy_multiplication_x:  |
| DDPG  | :heavy_multiplication_x:  | :heavy_check_mark:        |
| TD3   | :heavy_multiplication_x:  | :heavy_check_mark:        |
| SAC   | :heavy_multiplication_x:  | :heavy_check_mark:        |
| PPO   | :heavy_multiplication_x:  | :heavy_check_mark:        |

### Bandits
#### Multi-armed bandits
|                               | Discrete           | Continuous                |
| ---                           | :---:              | :---:                     |
| Pure Exploration              | :heavy_check_mark: | :heavy_multiplication_x:  |
| Epsilon Greedy                | :heavy_check_mark: | :heavy_multiplication_x:  |
| Thompson Sampling - Bernoulli | :heavy_check_mark: | :heavy_multiplication_x:  |
| Thompson Sampling - Gaussian  | :heavy_check_mark: | :heavy_multiplication_x:  |
| Upper Confidence Bounds (UCB) | :heavy_check_mark: | :heavy_multiplication_x:  |

#### Contextual bandits
|                          | Discrete                 | Continuous                |
| ---                      | :---:                    | :---:                     |
| Linear UCB               | :heavy_check_mark:       | :heavy_multiplication_x:  |
| Linear Thompson Sampling | :heavy_multiplication_x: | :heavy_multiplication_x:  |
| Neural-network approach  | :heavy_check_mark:       | :heavy_multiplication_x:  |

### Model-based learning
|                                       | Discrete                  | Continuous                |
| ---                                   | :---:                     | :---:                     |
| Dyna-Q                                | :heavy_check_mark:        | :heavy_multiplication_x:  |
| Deep Dyna-Q                           | :heavy_check_mark:        | :heavy_multiplication_x:  |
| Monte-Carlo Tree Search (MCTS)        | :heavy_check_mark:        | :heavy_multiplication_x:  |
| MB + Model Predictive Control         | :heavy_multiplication_x:  | :heavy_check_mark:        |
| Model-Based Policy Opitmization (MBPO)| :heavy_multiplication_x:  | :heavy_check_mark:        |

### Offline (batch) learning
|                                                    | Discrete                  | Continuous                |
| ---                                                | :---:                     | :---:                     |
| Conservative Q-learning (CQL)                      | :heavy_check_mark:        | :heavy_multiplication_x:  |
| Model-Based Offline Reinforcement Learning (MOReL) | :heavy_check_mark:        | :heavy_multiplication_x:  |
| Model-Based Offline Policy Optimization (MOPO)     | :heavy_multiplication_x:  | :heavy_check_mark:        |

### Other
|                    | Discrete           | Continuous                |
| ---                | :---:              | :---:                     |
| Behavioral Cloning | :heavy_check_mark: | :heavy_multiplication_x:  |
| Imitation Learning | :heavy_check_mark: | :heavy_multiplication_x:  |

## Installation
- Make sure you have Python 3.7 or higher installed
- Clone the repo
```
# Clone repo from github
git clone --depth 1 https://github.com/natetsang/open-rl

# Navigate to root folder
cd open-rl
```
- Create a virtual environment (Windows 10). Showing instructions from `virtualenv` but there are other options too!
```
# If not already installed, you might need to run this next line
pip install virtualenv 

# Create virtual environment called 'venv' in the root of the project
virtualenv venv

# Activate environment
venv\Scripts\activate
```
- Download requirements
```
pip install -r requirements.txt
```

## Contributing
If you're interested in contributing to `open-rl`, please fork the repo and make a pull request. Any support
is much appreciated!

## Citation
If you use this code, please cite it as follows:
```
@misc{Open-RL,
author = {Tsang, Nate},
title = {{Open-RL: Minimalistic implementations of reinforcment learning algorithms}},
url = {https://github.com/natetsang/open-rl},
year = {2021}
}
```

## Acknowledgements
This repo would not be possible without the following (tremendous) resources, which were relied upon heavily when
learning RL. I highly recommend going through these to learn more.
* [CS285](http://rail.eecs.berkeley.edu/deeprlcourse/) @ UC Berkeley - taught by Sergey Levine
* [Grokking Deep RL book](https://www.manning.com/books/grokking-deep-reinforcement-learning) by [@mimoralea](https://github.com/mimoralea/gdrl)
* More to be added soon!


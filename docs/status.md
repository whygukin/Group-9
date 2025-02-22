---
layout: default
title: Status
---

## Summary of the Project (updated)
For this project, we would like to train an agent that will be able to play multi-agent poker games such as limited Texas hold'em and perform better than fixed policy agents (such as ones that always choose a random action or always chooses to call). We will take into account exploitability, round win-rate, and total earnings as metrics when considering an agent's performance. We will use Proximal Policy Optimization (PPO) to train the agent, because it doesn't require complete knowledge of a game state and is able to learn a strategy directly; improving policies iteratively. 

## Approach
The main algorithms we will be using is PPO and DQN, with the goal of seeing how an on-policy algorithm like PPO would compare with an off-policy algorithm like DQN. We train two agents, one with each algorithm, and have them play themselves (self-play) in order to learn. For PPO, pure self-play wouldn't converge to to Nash equilibrium, but we hope that it will still lead to strong agent that could compete against fixed-policy agents. DQN, on the other hand, could approximate a Nash equilibrium by using it as the RL learner for Neural Fictitious Self-Play (NFSP), and having the agent train a reservoir buffer to learn an average strategy. 
So far... 

## Evaluation


## Remaining Goals and Challenges



## Resources Used
We used OpenSpiel's universal_poker implementation as a starting point for our code, and are using its implemented algorithms as a library. We also referenced Stanford class project's paper (https://web.stanford.edu/class/aa228/reports/2018/final96.pdf) when considering our approach to our project. 

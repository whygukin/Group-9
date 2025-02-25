---
layout: default
title: Status
---

## Summary of the Project (UPDATED)

Our inital proposal of the project was tailored to blackjack and using DQN to train. However we realized the sheer limitations of the project in a single agent game with only 2 cards. Instead we wanted to expand and play more complicated games such as poker. For this project, we would like to train an agent that can play multi-agent poker games like limited Texas hold'em. Our inital goal is to make sure we train the agents so that they are able to play poker using reinforcement learning algorithms like CFR and PPO. We initally thought of DQN however, the agent do not know all possible states making it difficult to justify using DQN. We did some research and found that potentially CFR might be able to aid in solving poker. PPO similarly can handle games that are partially observable, making a potential candidate. We realize that there are probably better algorithms out there, but these are the two algorithms we are focusing on. For the algorithms we will take into account such as exploitability, round win-rates, and total chip winnings as part of the agent's performance.

Our ultimate goal in this first inital proposal was to get a feel for the different algorithms and make sure they overall work. By work, I mean they are able to play poker and become somewhat successful at it.

## Approach
The approach we are taking to tackle the project is by using the algorithms mentioned: PPO and CFR. [INPUT WHAT CFR IS HERE]. On the other hand PPO (Policy Promixity Optimization) algorithm aims to reduce the surrogate loss function found in the policy in order to maxmize the rewards from the user. PPO was developed and researched at OpenAI and in their documentation, PPO has a lost function that looks like this:

$$
L(s,a, \theta_k, \theta) = \min \left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a), \ g(\epsilon, A^{\pi_{\theta_k}}(s,a)) \right),
$$

where  

$$
g(\epsilon, A) =
\begin{cases} 
(1+\epsilon)A & A \geq 0 \\
(1-\epsilon)A & A < 0.
\end{cases}
$$

## Evaluation


## Remaining Goals and Challenges

Now that we know our models are at least working to some extent, we aim to finish off the quarter/project by: 

- Compare the different algorithms that were used (CFR/PPO) and see their performance using similar parameters
- Training the agents that do not use random actions
- Agent vs Agents and see how they perform against each other with different parameters
- Analyze each of the agents and see what types of frequent actions they are taking, then optimize accordingly
- Figure out which is the more optimal algorithm for poker against these algorithms

This list might not be exhaustive and all of these goals are not expected to be completed due to the sheer limitation in time we have left for this project. However, we will strive our best to achieve as much
of these goals on this list as possible.

## Resources Used
We used OpenSpiel's universal_poker implementation as a starting point for our code, and are using its implemented algorithms as a library. We also referenced Stanford class project's paper (https://web.stanford.edu/class/aa228/reports/2018/final96.pdf) when considering our approach to our project. 

Helped with understanding a bit about the PPO algorithm (along with the lecture slides).
https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8

The PPO algorithm used based on the article above, but with some minor tweaks/changes.
https://github.com/ericyangyu/PPO-for-Beginners/blob/master/ppo.py  (repository - PPO for beginners)

PPO algorithm such as all the math from Open AI (developers themselves)
https://spinningup.openai.com/en/latest/algorithms/ppo.html 

Open-Spiel Environment (Specifically universal_poker environment).
https://github.com/google-deepmind/open_spiel 

Helped with integrating open-spiel environment with gym environment to use with the PPO algorithm
https://www.gymlibrary.dev/api/wrappers/ - help with using the algorithm

AI (ChatGPT) was used to help make some changes in the PPO algorithm (in the codebase above), such as helping change the PPO algorithm handle continuous to discrete inputs/outputs.

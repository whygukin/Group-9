---
layout: default
title: Status
---

## Summary of the Project (UPDATED)

Our inital proposal of the project was tailored to blackjack and using DQN to train. However we realized the sheer limitations of the project in a single agent game with only 2 cards. Instead we wanted to expand and play more complicated games such as poker. For this project, we would like to train an agent that can play multi-agent poker games like limited Texas hold'em. Our inital goal is to make sure we train the agents so that they are able to play poker using reinforcement learning algorithms like CFR and PPO. We initally thought of DQN however, the agent do not know all possible states making it difficult to justify using DQN. We did some research and found that potentially CFR might be able to aid in solving poker. PPO similarly can handle games that are partially observable, making a potential candidate. We realize that there are probably better algorithms out there, but these are the two algorithms we are focusing on. For the algorithms we will take into account such as exploitability, round win-rates, and total chip winnings as part of the agent's performance.

Our ultimate goal in this first inital proposal was to get a feel for the different algorithms and make sure they overall work. By work, I mean they are able to play poker and become somewhat successful at it.

## Approach
One of the aforementioned approaches we are taking to tackle the project is by using the PPO (Proximal Policy Optimization) algorithm created and developed by OpenAI. A variation of the actor-critic model, which helps the agent make better decisions based on the critic neural-network which influences the actor (which decisions to make) or vice versa. The PPO algorithm also aims to reduce the surrogate loss function found in the policy in order to maxmize the rewards from the user. PPO has a lost function that looks like this:

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

Basically finding the ratio of the policies multiplied by their advantage to see how it performs. Then, the algorithm compares it with the lower bound / upper bound changes the algorithm can perform. This makes it so the PPO algorithm cannot have a loss that is bigger / less than the desired amount (usually a small amount). This makes updating the policy incrementally small potentially taking longer to converge and or find a solution. However, it makes the algorithm much more stable in the long run.

This algorithm works well in Poker Reinforcement Learning because of its slow updating policies making it a stable algorithm. But also, it is capable of learning in partially observable environments.

The PPO algorithm was paired with the open-spiel environment that our TA recommended us for playing card games. The open-spiel environment also developed in part with google deepmind, contains universal poker which is a poker-like environment with adjustable parameters to reflect different variations of poker. For the PPO algorithm, we set the configurations as follows (similar to Texas Hold'em):

- Each player starts with 2000 chips
- 5 players total (1 as the designated agent to learn on PPO, 4 as complete random action agents)
- Each player inital bets are 50 chips
- 52 total cards in the deck (all 4 suits, 13 cards in each suit)
- 3 flipped cards in the beginning, 1 more each after. Maxes at 5. (as players continue to bet)
- Maxmium number total betting rounds cap at 4.

To learn more about how to play poker: https://bicyclecards.com/how-to-play/texas-holdem-poker.

The algorithm itself ran with these parameters:

- 1500000 timesteps
- 1600 max timesteps per episode
- 4800 timesteps per batch
- entropy coeffecient at 0.05
- clipping at 0.2
- gamma at 0.99
- learning rate at 0.001

Some of these paramters were created by the person who made the base PPO algorithm (which was modified later). The 4800 max timesteps per batch, 1600 max timesteps per episode were from the original code. Clipping, gamma, learning rate, entropy were all changed afterwards by playing around with the results of the algorithm.

Another algorithm we looked at was Neural Fictitious Self-Play, using DQN as the inner-RL algorithm. As Johannes Heinrich and David Silver describes it in their paper "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games", "NFSP combines FSP with neural network function approximation...  all players of
the game are controlled by separate NFSP agents that learn from simultaneous play against each other, i.e. self-play. An NFSP agent interacts with its fellow agents and memorizes its experience of game transitions and its own best response behaviour in two memories," and "treats these memories as two distinct datasets suitable for deep reinforcement learning and supervised classification respectively." (Heinrich and Silver) The agent trains a neural network to predict action values using DQN, resulting in a network that represents the agent's approximate best response strategy, which selects a random action with probability and otherwise chooses the action that maximizes the predicted action values. 

The agents were configured to have the following parameters: 
        # parameters for NFSP
        "hidden_layers_sizes": [256, 256],
        "reservoir_buffer_capacity": int(2e6),
        "anticipatory_param": 0.1,
        "batch_size": 256,
        "rl_learning_rate": 0.001,
        "sl_learning_rate": 0.001,
        "min_buffer_size_to_learn": 10000,
        "learn_every": 64,
        "optimizer_str": "adam",

        # parameters for DQN
        "replay_buffer_capacity": int(2e6),
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
and these were all given to the python implementation of the NFSP algorithm that OpenSpiel has available. 
 

## Evaluation

The results of the PPO algorithm is limited to 1500000 timesteps and the environment and the results may not reflect actual poker gameplay.
[INSERT GRAPH?]
The results of the graph that was generated from the performance of the agent using the PPO algorithm can be seen above. While the graph is hard to read, there is a clear trend in performance in average episodic return being higher as more iterations are performed while playing Texas hold em. The return seems to be steadily increasing (the reward / # of chips average gained per episode vs iteration). At around 30 iterations to 130 iterations, there seems to be some stagnation in terms of episodic reward growth. A possibility for that is due to the AI agent finding difficulty in trying to find other optimal strategies besides folding, thus leading to a -50 (which is the initial amount bet) and instantly folding. To combat this, we decided to add a way for the agent to explore a little bit to find other optimal strategies besides folding. After the stagnation in the initial episodic returns, there seems to be another gain all the way to touching 400 chips in one episode. Thus showing potential growth from the AI agent itself.

Another metric that was analyzed closely was the average actor loss. In order for the algorithm to be “stable”, we want the “surrogate loss function” to stabilize and or having a small error. This shows that the algorithm is likely to be converging which can be shown in the graph above. Stablizing at around -0.2 actor loss then exploring new strategies caused the AI agent to stabilize at a new -0.10 range of actor loss. This means that the PPO algorithm is likely to be improving over the iterations, while making small incremental changes to its policy.

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

Entropy in PPO algorithm:
https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe

Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (Johannes Heinrich and David Silver)
https://arxiv.org/pdf/1603.01121

AI (ChatGPT) was used to help make changes in the PPO algorithm (in the codebase above), 1) helping change the PPO algorithm handle continuous to discrete inputs/outputs. 2) Alter it so that it is able to add some sort of exploration within the PPO algorithm using entropy.

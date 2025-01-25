---
layout: default
title: Proposal
---

## Summary of the Project
For this project, we would like to train an agent that will be able to play Blackjack with other (average-skilled) players and win at least 50% of hands. We will leave out any betting or risk calculation and instead focus on training it to make the best decision to win the hand given its current knowledge of the table. The agent will receive information about the current state of the game, such as the player’s cards, the dealer’s visible card, and the current available actions as input. Based on this input, it will produce an action (such as “hit,” “stand,” or “fold”). The goal is to maximize the agent’s winnings over many rounds of Blackjack by adjusting its strategy over time.  The setup involves the RLCard library’s built-in Blackjack environment (does not include the actual algorithm), which provides the agent with the current game state (such as the player’s hand, the dealer’s visible card, and the legal actions). Our baseline approach will be using a classic DQN and then explore how a Rainbow DQN might yield better performance. Some applications include creating more challenging opponents for practicing Blackjack skills or exploring optimal strategies that can be applied in online gaming platforms.

## AI/ML Algorithms
While none of these algorithms are going to be final, we aim to utilize a variety of AI techniques to enhance our agent. AI/ML algorithms include using a combination of model-free, off-policy reinforcement learning (specifically Deep Q-Learning) with a neural network function approximator to train and evaluate the blackjack agent. We will also try to implement both DQN and Rainbow DQN, and compare the accuracy of the blackjack agent for the off-policy reinforcement learning. We might also established a Rule-Based agent to ensure a baseline for our agent.

## Evaluation Plan

Our evaluation plan to ensure that we can assess the success of our project is to use the amount of games won using the agent as the player. The agent will play a series of games ranging from 1,000 - 10,000 games of blackjack to score as many wins as possible within those games. We will then measure the agent's win rate (percentage of hands won). As researched, the average human win rate in Blackjack is around 32%. The goal of our project is to at least beat this baseline, an try to get around a 50% win rate. (A moonshot / less realistic goal would be significantly higher like 60-70% win rate). Achieving this would indicate that the agents would perform better than most humans, suggesting that the project is successful. we could also measure the agent's average return (net winnings) over a large number of Blackjack hands, comparing the performance against the baseline for average-skilled players as well as a simple rule-based baseline (for example, always “hit” when the player’s total is below 17). 

For a sanity check, I’ll observe the agent’s behavior in simple toy scenarios, such as when the player’s total is very close to 21 or when the dealer is showing a certain critical card. By stepping through these scenarios manually, I can verify that the agent’s chosen actions make intuitive sense. I also plan to visualize the learned Q-values or policy mappings for a few selected states to see if the agent is assigning high values to sensible actions (like “stand” at a high total). 

Moreover, we will calculate the success rates of the Q-values (tracking future expected rewards) and policies over time to see if the agent is learning meaningfully and contributing to a higher percentage of success rate in order to make better decisions for future games. An instance of this is a sanity check where if the agent is potentially having a 20 in their hand, the more optimal solution will mostly involve the agent to not draw another card. We can also test other potential game configurations later down the road.

## Meet with Instructor
We plan to meet the instructor a couple times throughout the quarter. We have already planned to sign up for one meeting coming up on Tuesday 3:45pm - 4:00pm. Other date(s) and time(s) are unknown at the current moment. We will meet the requirements to meet with the instructor for the project as requested throughout the quarter.

## AI Tool Usage
We will document any usage of AI throughout the project as requested from the instructor.

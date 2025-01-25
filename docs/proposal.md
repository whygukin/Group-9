---
layout: default
title: Proposal
---

## Summary of the Project

The main idea behind our project is to be able to have an AI play card games such as blackjack (at least that is the aim of our current project). 
Blackjack is a common card game played in casinos and sometimes casually with friends. As such, we wanted to create an AI/agent that can automatically play blackjack for us.
Our goal is to maxmize probability (our goal: at least 50% of hands), to make the best possible choices given the constraints within the game. This can include making decision from the dealer's hand, the current state of the game, and information on our current hand (input). Then, we can maxmize the probability of winning our games compared to our opponents (output). Applications of this project may include creating advanced AI for those that want maxmize their chances of winning against other human players or creating bots (digital opponents) that have various spiked difficulties. The agent will have 3 decisions (to hit - draw another card, to stand - not draw, fold - give up (went over 21)). Our project might utilize a toolkit using as RLCard to generate a Blackjack environment (does not create the actual algorithms).

## AI/ML Algorithms

While none of these algorithms are going to be final, we aim to utilize a variety of AI techniques to enhance our agent. AI/ML algorithms include using a combination of model-free, off-policy reinforcement learning (specifically Deep Q-Learning) with a neural network function approximator to train and evaluate the blackjack agent. We will also try to implement both DQN and Rainbow DQN, and compare the accuracy of the blackjack agent for the off-policy reinforcement learning. We might also established a Rule-Based agent to ensure a baseline for our agent.

## Evaluation Plan

Our evaluation plan to ensure that we can assess the success of our project is to use the amount of games won using the agent as the player. The agent will play a series of games ranging from 1,000 - 10,000 games of blackjack to score as many wins as possible within those games. As researched, the average human win rate in Blackjack is around 32%. The goal of our project is to at least beat this baseline. (A moonshot / less realistic goal would be 50%). Achieving this would indicate that the agents would perform better than most humans, suggesting that the project is successful. For sanity checks, we will try to 

Moreover, we will calculate the success rates of the Q-values (tracking future expected rewards) and policies over time to see if the agent is learning meaningfully and contributing to a higher percentage of success rate in order to make better decisions for future games. An instance of this is a sanity check where if the agent is potentially having 20 in their hand, the more optimal solution will mostly involve the agent to not draw another card. We can also test other potential game configurations later down the road.

## Meet with Instructor
We plan to meet the instructor a couple times throughout the quarter. We have already planned to sign up for one meeting coming up on Tuesday 3:45pm - 4:00pm. Other date(s) and time(s) are unknown at the current moment.
We will meet the requirements to meet with the instructor for the project as requested throughout the quarter.

## AI Tool Usage
We will document any usage of AI throughout the project as requested from the instructor.

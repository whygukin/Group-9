import os
import matplotlib.pyplot as plt
import numpy as np
import time

import pyspiel

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import nfsp
from open_spiel.python.algorithms import random_agent

import gc

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()  # Disable TF2 eager execution

class CustomPokerEnv(rl_environment.Environment):
    def __init__(self, params):
        self._game = pyspiel.load_game("universal_poker", params)
        super().__init__(self._game)


def evaluate_against_random_bots(env, trained_agents, num_episodes=500):
    """Evaluates trained agents against random agents."""
    random_agents = [
        random_agent.RandomAgent(player_id=idx, num_actions=env.action_spec()["num_actions"])
        for idx in range(env.num_players)
    ]
    
    # Evaluate trained agents as player 0
    wins_as_player0 = 0
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            if player_id == 0:
                agent_output = trained_agents[0].step(time_step, is_evaluation=True)
                action = agent_output.action
            else:
                agent_output = random_agents[player_id].step(time_step)
                action = agent_output.action
            
            time_step = env.step([action])
        
        # Check if player 0 won
        if time_step.rewards[0] > 0:
            wins_as_player0 += 1
    
    # Evaluate trained agents as player 1
    wins_as_player1 = 0
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            if player_id == 1:
                agent_output = trained_agents[1].step(time_step, is_evaluation=True)
                action = agent_output.action
            else:
                agent_output = random_agents[player_id].step(time_step)
                action = agent_output.action
            
            time_step = env.step([action])
        
        # Check if player 1 won
        if time_step.rewards[1] > 0:
            wins_as_player1 += 1
    
    return wins_as_player0 / num_episodes, wins_as_player1 / num_episodes

def save_agents(agents, save_dir, episode):
    """Save trained agents to disk with episode number."""
    episode_dir = os.path.join(save_dir, f"episode_{episode}")
    os.makedirs(episode_dir, exist_ok=True)
    for idx, agent in enumerate(agents):
        agent.save(os.path.join(episode_dir, f"agent_{idx}"))
    print(f"Agents saved to {episode_dir}")

def write_to_log(save_dir, message, print_to_console=True):
    log_file = os.path.join(save_dir, "training_log.txt")
    with open(log_file, "a") as f:
        f.write(message + "\n")
    if print_to_console:
        print(message)


if __name__ == "__main__":
    print(gc.isenabled())
    
    # Create save directory (hard-coded)
    save_dir = "/home/ethanso/CS175/nfsp_1v1_models_adam/run_17"
    os.makedirs(save_dir, exist_ok=True)
    
    session = tf.Session()
    session.__enter__()  # Make the session active
    
    env = CustomPokerEnv({
        "betting": "limit",
        "numPlayers": 2,
        "numRounds": 2,
        "blind": "2 4",
        "raiseSize": "4 8",
        "firstPlayer": "1 1",
        "maxRaises": "2 2",
        "numSuits": 2,
        "numRanks": 5,
        "numHoleCards": 1,
        "numBoardCards": "0 2",
        "stack": "20 20"
    })

    num_players = 2
    
    agent_configs = {
        "hidden_layers_sizes": [256, 256],        
        "reservoir_buffer_capacity": int(1e6),    
        "anticipatory_param": 0.1,
        "batch_size": 256,
        "rl_learning_rate": 0.01,
        "sl_learning_rate": 0.01,
        "min_buffer_size_to_learn": 10000,
        "learn_every": 64,
        "optimizer_str": "adam",
        
        # parameters for DQN
        "replay_buffer_capacity": int(1e6),
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
    }
    
    # Initialize agents
    agents = [
        nfsp.NFSP(
            player_id=i,
            state_representation_size=env.observation_spec()["info_state"][0],
            num_actions=env.action_spec()["num_actions"],
            session=session,
            **agent_configs
        )
        for i in range(num_players)
    ]
    
    session.run(tf.global_variables_initializer())

    # Training parameters
    num_episodes = 400000
    eval_every = 10000
    save_every = 50000
    
    returns = []
    win_rates = []
    rl_losses_0 = []
    sl_losses_0 = []
    rl_losses_1 = []
    sl_losses_1 = []
    episode_numbers_0 = []
    episode_numbers_1 = []
    

    start_time = time.time()
    for episode in range(num_episodes):
        time_step = env.reset()
        episode_return = np.zeros(num_players)

        while not time_step.last():
            current_player = time_step.observations["current_player"]
            agent = agents[current_player]
            agent_output = agent.step(time_step)
            action = agent_output.action

            time_step = env.step([action])
            episode_return += np.array(time_step.rewards)

        # At the end of the episode, let agents observe the final state
        if time_step.last():
            for agent in agents:
                agent.step(time_step)
        
        returns.append(episode_return)

        # Periodic evaluation
        if episode % eval_every == 0 or episode == num_episodes - 1:
            elapsed_time = round(time.time() - start_time, 2)
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)

            for agent in agents:
                sl_loss, rl_loss = agent.loss
                if agent.player_id == 0:
                    if sl_loss is not None:
                        sl_losses_0.append(sl_loss)
                        rl_losses_0.append(rl_loss)
                        episode_numbers_0.append(episode)
                    else:
                        rl_losses_0.append(rl_loss)
                else:
                    if sl_loss is not None:
                        sl_losses_1.append(sl_loss)
                        rl_losses_1.append(rl_loss)
                        episode_numbers_1.append(episode)
                    else:
                        rl_losses_1.append(rl_loss)
                
            
            win_rate_p0, win_rate_p1 = evaluate_against_random_bots(env, agents)
            win_rates.append((win_rate_p0, win_rate_p1))

            log_message = f"""
-------------------- Episode #{episode} --------------------
Win rate as P0 against random bots: {win_rate_p0:.3f}
Win rate as P0 against random bots: {win_rate_p1:.3f}
Average return over last {eval_every} episodes: {np.mean(returns[-eval_every:], axis=0)}
Time elapsed: {minutes} mins and {seconds:02d} secs
------------------------------------------------------
"""
            write_to_log(save_dir, log_message)
            
        
        # Save agent
        if episode % save_every == 0 and episode > 0:
            elapsed_time = round(time.time() - start_time, 2)
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            
            save_agents(agents, save_dir, episode)
   
    # Final save
    save_agents(agents, save_dir, num_episodes)

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # plot win-rate against uniform random policy
    min_len = min(len(np.arange(0, num_episodes, eval_every)), len(win_rates))
    x_values = np.arange(0, num_episodes, eval_every)[:min_len]
    win_rates_truncated = win_rates[:min_len] 
    axes[0, 0].plot(x_values, [wr[0] for wr in win_rates_truncated], label='Player 0')
    axes[0, 0].plot(x_values, [wr[1] for wr in win_rates_truncated], label='Player 1')
    axes[0, 0].set_xlabel("Episodes")
    axes[0, 0].set_ylabel("Win Rate vs Random")
    axes[0, 0].legend()
    axes[0, 0].set_title("Win Rate Against Random Bot")
   
    # plot average return of the agent (payoff per episode)
    returns_p0 = [r[0] for r in returns] 
    returns_p1 = [r[1] for r in returns] 
    axes[0, 1].plot(np.convolve(returns_p0, np.ones(1000)/1000, mode='valid'), label='Player 0')
    axes[0, 1].plot(np.convolve(returns_p1, np.ones(1000)/1000, mode='valid'), label='Player 1')
    axes[0, 1].set_xlabel("Episodes")
    axes[0, 1].set_ylabel("Average Return (Moving Avg)")
    axes[0, 1].legend()
    axes[0, 1].set_title("Training Returns")

    # plot reinforcement learning loss
    rl_losses_0_truncated = rl_losses_0[:min_len]
    rl_losses_1_truncated = rl_losses_1[:min_len]
    axes[1, 0].plot(x_values, rl_losses_0_truncated, label='Player 0')
    axes[1, 0].plot(x_values, rl_losses_1_truncated, label='Player 1')
    axes[1, 0].set_xlabel("Episodes")
    axes[1, 0].set_ylabel("RL Loss")
    axes[1, 0].legend()
    axes[1, 0].set_title("Loss in RL Portion")

    # plot supervised learning loss 
    axes[1, 1].plot(episode_numbers_0, sl_losses_0, label='Player 0')
    axes[1, 1].plot(episode_numbers_1, sl_losses_1, label='Player 1')
    axes[1, 1].set_xlabel("Episodes")
    axes[1, 1].set_ylabel("SL Loss")
    axes[1, 1].legend()
    axes[1, 1].set_title("Loss in SL Portion")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_results_final.png")

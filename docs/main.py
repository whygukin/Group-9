import os
import matplotlib.pyplot as plt
import numpy as np
import time

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import nfsp
from open_spiel.python.algorithms import random_agent

import gc

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()  # Disable TF2 eager execution

def evaluate_against_random_bots(env, trained_agents, num_episodes=1000):
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

def load_agents(session, env, save_dir, agent_configs):
    """Load trained agents from disk."""
    num_players = env.num_players
    agents = []
    for i in range(num_players):
        agent = nfsp.NFSP(
            player_id=i,
            state_representation_size=env.observation_spec()["info_state"][0],
            num_actions=env.action_spec()["num_actions"],
            session=session,
            **agent_configs
        )
        agent.restore(save_dir + f"/agent_{i}")
        agents.append(agent)
    return agents

# def load_all_returns(save_dir, current_episode):
    # """Load all saved returns and combine them with current returns."""
    # all_returns = []
    
    # # Load saved returns from files
    # for ep in range(10000, current_episode, 10000):
    #     try:
    #         saved_returns = np.load(f"{save_dir}/returns_{ep}.npy")
    #         all_returns.extend(saved_returns)
    #     except FileNotFoundError:
    #         print(f"Warning: Could not find returns file for episode {ep}")
    
    # # Add current returns that haven't been saved yet
    # all_returns.extend(returns)
    
    # return all_returns

def plot_returns_from_files(save_dir, current_episode, returns, save_count, window_size=1000):
    """Plot returns data from saved files without loading all data at once."""
    plt.figure(figsize=(12, 5))
    
    # Initialize arrays for the moving average calculation
    sum_p0 = np.zeros(window_size)
    sum_p1 = np.zeros(window_size)
    count = 0
    
    # Process each saved file
    for ep in range(10000, current_episode + 1, 10000):
        try:
            # Load one file at a time
            file_path = f"{save_dir}/returns_{ep}.npy"
            
            # Process the file in chunks to avoid memory issues
            for chunk in np.load(file_path, mmap_mode='r'):  # Memory-mapped mode
                # Update the moving average
                returns_p0 = chunk[0]  # Player 0 return
                returns_p1 = chunk[1]  # Player 1 return
                
                # Update the sums
                sum_p0 = np.roll(sum_p0, -1)
                sum_p1 = np.roll(sum_p1, -1)
                sum_p0[-1] = returns_p0
                sum_p1[-1] = returns_p1
                
                count += 1
                
                # Plot a point every 1000 episodes to avoid overcrowding
                if count % 1000 == 0:
                    plt.plot(count, np.mean(sum_p0), 'b.', alpha=0.5)
                    plt.plot(count, np.mean(sum_p1), 'r.', alpha=0.5)
                    
        except FileNotFoundError:
            print(f"Warning: Could not find returns file for episode {ep}")
    
    # Add current returns that haven't been saved yet
    for r in returns:
        sum_p0 = np.roll(sum_p0, -1)
        sum_p1 = np.roll(sum_p1, -1)
        sum_p0[-1] = r[0]
        sum_p1[-1] = r[1]
        count += 1
        
        if count % 1000 == 0:
            plt.plot(count, np.mean(sum_p0), 'b.', alpha=0.5)
            plt.plot(count, np.mean(sum_p1), 'r.', alpha=0.5)
    
    # Add smoothed lines
    plt.plot([], [], 'b-', label='Player 0')
    plt.plot([], [], 'r-', label='Player 1')
    
    plt.xlabel("Episodes")
    plt.ylabel("Average Return (Moving Avg)")
    plt.legend()
    plt.title("Training Returns")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_results_{save_count}.png")
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    print(gc.isenabled())
    # Create save directory
    save_dir = "./poker_nfsp_models"
    os.makedirs(save_dir, exist_ok=True)
    
    session = tf.Session()
    session.__enter__()  # Make the session active

    env = rl_environment.Environment(
        "universal_poker", 
        {"numPlayers": 2,
        "stack": "2000 2000",
        "blind": "50 50",
        "numRanks": 13,
        "numHoleCards": 2,
        "numBoardCards": "0 3 1 1",
        "numRounds": 4,
        "bettingAbstraction": "fullgame"}
    )
    num_players = env.num_players
    
    agent_configs = {
        "hidden_layers_sizes": [128, 128],        # 
        "reservoir_buffer_capacity": int(5e5),    # 
        "anticipatory_param": 0.1,                # Probability of using best response
        "batch_size": 256,                        # Larger batch size
        "rl_learning_rate": 0.001,                # 
        "sl_learning_rate": 0.001,                # 
        "min_buffer_size_to_learn": 10000,        # Wait for more experience before learning
        "learn_every": 64,                        # Learn frequency
        "optimizer_str": "adam",                  # Adam optimizer
        
        # parameters for DQN
        "replay_buffer_capacity": int(5e5),       # Experience replay capacity
        "epsilon_start": 0.06,                    # Starting exploration rate
        "epsilon_end": 0.001,                     # Final exploration rate
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
    num_episodes = 1000000
    eval_every = 10000
    save_every = 100000
    
    returns = []
    eval_returns = []
    win_rates = []

    save_count = 0

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
            
            win_rate_p0, win_rate_p1 = evaluate_against_random_bots(env, agents)
            win_rates.append((win_rate_p0, win_rate_p1))

            print()
            print(f"-------------------- Episode #{episode} --------------------", flush=True)
            print(f"Win rate as P0 against random bots: {win_rate_p0:.3f}", flush=True)
            print(f"Win rate as P1 against random bots: {win_rate_p1:.3f}", flush=True)
            print(f"Average return over last {eval_every} episodes: {np.mean(returns[-eval_every:], axis=0)}", flush=True)
            print(f"Time elapsed: {minutes} mins and {seconds:02d} secs", flush=True)
            print(f"------------------------------------------------------")
            print()
            
            # Save returns to disk
            np.save(f"{save_dir}/returns_{episode}.npy", np.array(returns))
            # Clear the list to free memory
            returns = []
            
        
        # Plot data periodically
        if episode % save_every == 0 and episode > 0:
            save_count += 1
            # Load all returns data
            plot_returns_from_files(save_dir, episode, returns, save_count, window_size=1000)
            # Plot results
            # plt.figure(figsize=(12, 5))
            # plt.subplot(1, 2, 1)
            # returns_p0 = [r[0] for r in returns]
            # returns_p1 = [r[1] for r in returns]
            # plt.plot(np.convolve(returns_p0, np.ones(1000)/1000, mode='valid'), label='Player 0')
            # plt.plot(np.convolve(returns_p1, np.ones(1000)/1000, mode='valid'), label='Player 1')
            # plt.xlabel("Episodes")
            # plt.ylabel("Average Return (Moving Avg)")
            # plt.legend()
            # plt.title("Training Returns")
            
            # plt.tight_layout()
            # plt.savefig(f"{save_dir}/training_results_{save_count}.png")
    
   
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)

    min_len = min(len(np.arange(0, num_episodes, eval_every)), len(win_rates))
    x_values = np.arange(0, num_episodes, eval_every)[:min_len]
    win_rates_truncated = win_rates[:min_len]
    plt.plot(x_values, [wr[0] for wr in win_rates_truncated], label='Player 0')
    plt.plot(x_values, [wr[1] for wr in win_rates_truncated], label='Player 1')

    plt.xlabel("Episodes")
    plt.ylabel("Win Rate vs Random")
    plt.legend()
    plt.title("Win Rate Against Random Bot")
    
    plt.subplot(1, 2, 2)
    returns_p0 = [r[0] for r in returns]
    returns_p1 = [r[1] for r in returns]
    plt.plot(np.convolve(returns_p0, np.ones(1000)/1000, mode='valid'), label='Player 0')
    plt.plot(np.convolve(returns_p1, np.ones(1000)/1000, mode='valid'), label='Player 1')
    plt.xlabel("Episodes")
    plt.ylabel("Average Return (Moving Avg)")
    plt.legend()
    plt.title("Training Returns")
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_results_final.png")
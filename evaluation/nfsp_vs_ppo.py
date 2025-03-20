import os
import matplotlib.pyplot as plt
import numpy as np
import time

import pyspiel
import gc

from open_spiel.python import rl_environment
from open_spiel.python import policy
from open_spiel.python.algorithms import nfsp
from open_spiel.python.algorithms.nfsp import MODE
from open_spiel.python.bots import policy as bot_policy

from stable_baselines3 import PPO


import tensorflow.compat.v1 as tf1
tf1.disable_eager_execution()  # Disable TF2 eager execution

class CustomPokerEnv(rl_environment.Environment):
    def __init__(self, params):
        self._game = pyspiel.load_game("universal_poker", params)
        super().__init__(self._game)

class NFSPPolicy(policy.Policy):
    """NFSP policy wrapper."""

    def __init__(self, nfsp_agent, game):
        """Initialize the NFSP policy wrapper."""
        self._nfsp_agent = nfsp_agent
        self._game = game

    def action_probabilities(self, state, player_id=None):
        """Returns action probabilities for the given state."""
        # Get legal actions
        legal_actions = state.legal_actions()
        if not legal_actions:
            return {}

        # Get info state
        info_state = np.array(state.information_state_tensor(self._nfsp_agent.player_id))
        
        # Use the NFSP agent's _act method to get action probabilities
        with self._nfsp_agent.temp_mode_as(MODE.average_policy):
            _, probs = self._nfsp_agent._act(info_state, legal_actions)
        
        # Convert to dictionary mapping actions to probabilities
        action_probs = {action: probs[action] for action in legal_actions}
        return action_probs
    

class PPOPolicy:
    """Wrapper class to convert a Stable Baselines 3 PPO model to an OpenSpiel policy."""

    def __init__(self, game, player_id, ppo_model):
        """Initialize the policy wrapper.
        
        Args:
            game: The OpenSpiel game object
            player_id: The player this policy is for
            ppo_model: The loaded Stable Baselines 3 PPO model
        """
        self._game = game
        self._player_id = player_id
        self._ppo_model = ppo_model

    def action_probabilities(self, state, player_id=None):
        """Returns a dictionary of action probabilities for the given state.
        
        Args:
            state: A `pyspiel.State` object.
            player_id: Optional, the player id for which we want the action probabilities.
                If None, the player id will be the one stored in the class.
        
        Returns:
            A dictionary mapping action IDs to probabilities.
        """

        def state_to_observation(state, player_id):
            """Convert an OpenSpiel state to the observation format expected by the PPO model.
            
            This function needs to be customized based on how your PPO model was trained.
            """
            # Example implementation - you'll need to adjust this based on your specific model
            if hasattr(state, "information_state_tensor"):
                return np.array(state.information_state_tensor(player_id))
            elif hasattr(state, "observation_tensor"):
                return np.array(state.observation_tensor(player_id))
            else:
                raise ValueError("State doesn't provide tensor observations")
    
        if player_id is None:
            player_id = self._player_id
        
        if state.current_player() != player_id:
            return {a: 1.0 / len(state.legal_actions(player_id)) 
                   for a in state.legal_actions(player_id)}
        
        # Convert the state to the format expected by the PPO model
        obs = state_to_observation(state, player_id)  # You need to implement this function
        
        # Get action from the model
        action, _states = self._ppo_model.predict(obs, deterministic=False)
        
        # If your model outputs distributions directly:
        # action_dist = self._ppo_model.policy.get_distribution(obs)
        # probs = action_dist.distribution.probs.detach().numpy()
        
        # For now, we'll create a simple distribution with all probability on the chosen action
        legal_actions = state.legal_actions(player_id)
        probs = {a: 0.0 for a in legal_actions}
        
        # Ensure the action is in legal_actions
        if action in legal_actions:
            # probs[action] = 1.0 # causing TypeError: unhashable type: 'numpy.ndarray' 
            # Convert NumPy array to a Python int
            action_int = int(action)
            probs[action_int] = 1.0
        else:
            # If the model outputs an illegal action, distribute uniformly
            for a in legal_actions:
                probs[a] = 1.0 / len(legal_actions)
        
        return probs

def load_nfsp_agent(session, env, save_dir, agent_configs, episode, player_id):
    """Load trained nfsp agent from disk for a specific episode."""
    episode_dir = os.path.join(save_dir, f"episode_{episode}")
    agent_path = os.path.join(episode_dir, f"agent_{player_id}")
       
    agent = nfsp.NFSP(
            player_id=player_id,
            state_representation_size=env.observation_spec()["info_state"][0],
            num_actions=env.action_spec()["num_actions"],
            session=session,
            **agent_configs
    )

    agent.restore(agent_path)
    
    return agent

def write_to_log(save_dir, message, print_to_console=True):
    log_file = os.path.join(save_dir, "training_log.txt")
    with open(log_file, "a") as f:
        f.write(message + "\n")
    if print_to_console:
        print(message)

def play_games(game, bots, num_games):
    """Play multiple games and return the results."""
    returns = []
    for _ in range(num_games):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                # Sample a chance event
                outcomes = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                # The current player makes a move
                current_player = state.current_player()
                bot = bots[current_player]
                action = bot.step(state)
                state.apply_action(action)
        returns.append(state.returns())
    return returns



if __name__ == "__main__":
    print(gc.isenabled())
    session = tf1.Session()
    session.__enter__()  # Make the session active

    CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF = """
GAMEDEF
limit
numPlayers = 2
numRounds = 2
blind = 2 4
raiseSize = 4 8
firstPlayer = 1
maxRaises = 2 2
numSuits = 2
numRanks = 5
numHoleCards = 1
numBoardCards = 0 2
stack = 20
END GAMEDEF
"""
    game = pyspiel.universal_poker.load_universal_poker_from_acpc_gamedef(CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF)

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

    nfsp_save_dir = "/home/ethanso/CS175/nfsp_1v1_models_adam/run_17"
    agent_configs = {
        "hidden_layers_sizes": [256, 256],        
        "reservoir_buffer_capacity": int(1e6),    
        "anticipatory_param": 0.1,                # Probability of using best response
        "batch_size": 256,
        "rl_learning_rate": 0.01,
        "sl_learning_rate": 0.01,
        "min_buffer_size_to_learn": 10000,        # Wait for more experience before learning
        "learn_every": 64,                        # Learn frequency
        "optimizer_str": "adam",                  # Adam optimizer
        
        # parameters for DQN
        "replay_buffer_capacity": int(1e6),       # Experience replay capacity
        "epsilon_start": 1.0,                     # Starting exploration rate
        "epsilon_end": 0.01,                      # Final exploration rate
    }

    session.run(tf1.global_variables_initializer())

    nfsp_agent_0 = load_nfsp_agent(session, env, nfsp_save_dir, agent_configs, 500000, player_id=0)
    nfsp_agent_1 = load_nfsp_agent(session, env, nfsp_save_dir, agent_configs, 500000, player_id=1)
    ppo_loaded_model = PPO.load("ppo_poker_model_final")

    # Create policy objects
    nfsp_policy_0 = NFSPPolicy(nfsp_agent_0, game)
    nfsp_policy_1 = NFSPPolicy(nfsp_agent_1, game)

    # Create policy bots
    rng = np.random.RandomState()
    nfsp_bot_0 = bot_policy.PolicyBot(0, rng, nfsp_policy_1)
    nfsp_bot_1 = bot_policy.PolicyBot(1, rng, nfsp_policy_1)

    ppo_bot_0 = bot_policy.PolicyBot(0, rng, PPOPolicy(env.game, player_id=0, ppo_model=ppo_loaded_model))
    ppo_bot_1 = bot_policy.PolicyBot(1, rng, PPOPolicy(env.game, player_id=1, ppo_model=ppo_loaded_model))

    nfsp_ppo_bots = [nfsp_bot_0, ppo_bot_1]
    ppo_nfsp_bots = [ppo_bot_0, nfsp_bot_1]

    start_time = time.time()


    num_iterations = 10  # Number of iterations
    num_games = 1000  # Number of games per iteration

    nvc_nfsp_win_rates = []
    nvc_ppo_win_rates = []
    nvc_draw_rates = []

    cvn_nfsp_win_rates = []
    cvn_ppo_win_rates = []
    cvn_draw_rates = []

    for iteration in range(num_iterations):
        nvc_nfsp_wins = 0
        nvc_ppo_wins = 0
        nvc_draws = 0
        cvn_nfsp_wins = 0
        cvn_ppo_wins = 0
        cvn_draws = 0
    
        all_returns_nfsp_ppo = play_games(env.game, nfsp_ppo_bots, num_games)
        all_returns_ppo_nfsp = play_games(env.game, ppo_nfsp_bots, num_games)

        # Count wins
        nvc_nfsp_wins = sum(1 for r in all_returns_nfsp_ppo if r[0] > r[1])
        nvc_ppo_wins = sum(1 for r in all_returns_nfsp_ppo if r[1] > r[0])
        nvc_draws = sum(1 for r in all_returns_nfsp_ppo if r[0] == r[1])

        cvn_ppo_wins = sum(1 for r in all_returns_ppo_nfsp if r[0] > r[1])
        cvn_nfsp_wins = sum(1 for r in all_returns_ppo_nfsp if r[1] > r[0])
        cvn_draws = sum(1 for r in all_returns_ppo_nfsp if r[0] == r[1])

        # Calculate win rates
        nvc_nfsp_win_rate = nvc_nfsp_wins / num_games * 100
        nvc_ppo_win_rate = nvc_ppo_wins / num_games * 100
        nvc_draw_rate = nvc_draws / num_games * 100

        nvc_nfsp_win_rates.append(nvc_nfsp_win_rate)
        nvc_ppo_win_rates.append(nvc_ppo_win_rate)
        nvc_draw_rates.append(nvc_draw_rate)

        cvn_nfsp_win_rate = cvn_nfsp_wins / num_games * 100
        cvn_ppo_win_rate = cvn_ppo_wins / num_games * 100
        cvn_draw_rate = cvn_draws / num_games * 100

        cvn_nfsp_win_rates.append(cvn_nfsp_win_rate)
        cvn_ppo_win_rates.append(cvn_ppo_win_rate)
        cvn_draw_rates.append(cvn_draw_rate)

        elapsed_time = round(time.time() - start_time, 2)
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        log_message = f"""
NFSP vs PPO results after {num_games*(iteration+1)} games:
NFSP wins: {nvc_nfsp_wins} ({nvc_nfsp_wins/num_games*100:.1f}%)
PPO wins: {nvc_ppo_wins} ({nvc_ppo_wins/num_games*100:.1f}%)
Draws: {nvc_draws} ({nvc_draws/num_games*100:.1f}%)
----------------------------------------------------------------
PPO vs NFSP results after {num_games*(iteration+1)} games:
NFSP wins: {cvn_nfsp_wins} ({cvn_nfsp_wins/num_games*100:.1f}%)
PPO wins: {cvn_ppo_wins} ({cvn_ppo_wins/num_games*100:.1f}%)
Draws: {cvn_draws} ({cvn_draws/num_games*100:.1f}%)

Time elapsed: {minutes} mins and {seconds:02d} secs
"""
        write_to_log("/home/ethanso/CS175/comparisons/nfsp_vs_ppo_1", log_message) # run_6 fixes potential flipped player ids 
                                                                           # run 8 fixes the issue of using the wrong nfsp agent
                                                                            # run 10 is correctly loaded nfsp agents
                                                                            # run 11 is nfsp vs ppo
                                                                        

    # After all iterations, create and save the graph
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    x = range(1, num_iterations + 1)
    plt.plot(x, nvc_nfsp_win_rates, label='NFSP')
    plt.plot(x, nvc_ppo_win_rates, label='PPO')
    plt.plot(x, nvc_draw_rates, label='Draws')
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate (%)')
    plt.title('NFSP vs PPO Performance Over Iterations')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    x = range(1, num_iterations + 1)
    plt.plot(x, cvn_nfsp_win_rates, label='NFSP')
    plt.plot(x, cvn_ppo_win_rates, label='PPO')
    plt.plot(x, cvn_draw_rates, label='Draws')
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate (%)')
    plt.title('PPO vs NFSP Performance Over Iterations')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the plot
    plt.savefig('/home/ethanso/CS175/comparisons/nfsp_vs_ppo_1/performance_graph.png')
    plt.close()





    # Clean up
    session.__exit__(None, None, None)


    # source venv/bin/activate
    # python3 nfsp_vs_ppo.py
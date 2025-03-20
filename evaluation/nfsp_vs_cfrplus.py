import os
import matplotlib.pyplot as plt
import numpy as np
import time

import pyspiel
import gc
import pickle

from open_spiel.python import rl_environment
from open_spiel.python import policy
from open_spiel.python.algorithms import nfsp
from open_spiel.python.algorithms.nfsp import MODE
from open_spiel.python.bots import policy as bot_policy

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

def load_nfsp_agent(session, env, save_dir, agent_configs, episode, player_id):
    """Load trained nfsp agent from disk for a specific episode."""
    episode_dir = os.path.join(save_dir, f"episode_{episode}")
    agent_path = os.path.join(episode_dir, f"agent_{player_id}")
    
    # Check if there's a .meta file in the directory
    # meta_files = [f for f in os.listdir(agent_path) if f.endswith('.meta')]
    # if not meta_files:
    #     raise ValueError(f"No .meta files found in {agent_path}")
    
    # Use the first .meta file found
    # meta_file = meta_files[0]
    # checkpoint_path = os.path.join(agent_path, meta_file[:-5])  # Remove .meta extension
    
    agent = nfsp.NFSP(
            player_id=player_id,
            state_representation_size=env.observation_spec()["info_state"][0],
            num_actions=env.action_spec()["num_actions"],
            session=session,
            **agent_configs
    )

    agent.restore(agent_path)
    return agent

def load_cfr_solver(save_dir, iteration):
    """Load CFR solver state from disk."""
    iteration_dir = os.path.join(save_dir, f"iteration_{iteration}")
    # Load the solver directly
    with open(os.path.join(iteration_dir, "solver.pkl"), "rb") as f:
        solver = pickle.load(f)
    print(f"CFRPlus solver loaded from {iteration_dir}")
    return solver

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
    cfrplus_save_dir = "/home/ethanso/CS175/cfrplus_1v1_models/run_3"

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

    cfr_solver = load_cfr_solver(cfrplus_save_dir, 500)

    session.run(tf1.global_variables_initializer())

    nfsp_agent_0 = load_nfsp_agent(session, env, nfsp_save_dir, agent_configs, 500000, player_id=0)
    nfsp_agent_1 = load_nfsp_agent(session, env, nfsp_save_dir, agent_configs, 500000, player_id=1)

    # Create policy objects
    nfsp_policy_0 = NFSPPolicy(nfsp_agent_0, game)
    nfsp_policy_1 = NFSPPolicy(nfsp_agent_1, game)
    cfr_policy = cfr_solver.average_policy()

    # Create policy bots
    rng = np.random.RandomState()
    nfsp_bot_0 = bot_policy.PolicyBot(0, rng, nfsp_policy_1)
    cfr_bot_1 = bot_policy.PolicyBot(1, rng, cfr_policy)

    nfsp_bot_1 = bot_policy.PolicyBot(1, rng, nfsp_policy_1)
    cfr_bot_0 = bot_policy.PolicyBot(0, rng, cfr_policy)

    nfsp_cfr_bots = [nfsp_bot_0, cfr_bot_1]
    cfr_nfsp_bots = [cfr_bot_0, nfsp_bot_1]

    start_time = time.time()


    num_iterations = 10  # Number of iterations
    num_games = 1000  # Number of games per iteration
    nvc_nfsp_win_rates = []
    nvc_cfr_win_rates = []
    nvc_draw_rates = []

    cvn_nfsp_win_rates = []
    cvn_cfr_win_rates = []
    cvn_draw_rates = []

    for iteration in range(num_iterations):
        nvc_nfsp_wins = 0
        nvc_cfr_wins = 0
        nvc_draws = 0
        cvn_nfsp_wins = 0
        cvn_cfr_wins = 0
        cvn_draws = 0
    
        all_returns_nfsp_cfr = play_games(env.game, nfsp_cfr_bots, num_games)
        all_returns_cfr_nfsp = play_games(env.game, cfr_nfsp_bots, num_games)

        # Count wins
        nvc_nfsp_wins = sum(1 for r in all_returns_nfsp_cfr if r[0] > r[1])
        nvc_cfr_wins = sum(1 for r in all_returns_nfsp_cfr if r[1] > r[0])
        nvc_draws = sum(1 for r in all_returns_nfsp_cfr if r[0] == r[1])

        cvn_cfr_wins = sum(1 for r in all_returns_cfr_nfsp if r[0] > r[1])
        cvn_nfsp_wins = sum(1 for r in all_returns_cfr_nfsp if r[1] > r[0])
        cvn_draws = sum(1 for r in all_returns_cfr_nfsp if r[0] == r[1])

        # Calculate win rates
        nvc_nfsp_win_rate = nvc_nfsp_wins / num_games * 100
        nvc_cfr_win_rate = nvc_cfr_wins / num_games * 100
        nvc_draw_rate = nvc_draws / num_games * 100

        nvc_nfsp_win_rates.append(nvc_nfsp_win_rate)
        nvc_cfr_win_rates.append(nvc_cfr_win_rate)
        nvc_draw_rates.append(nvc_draw_rate)

        cvn_nfsp_win_rate = cvn_nfsp_wins / num_games * 100
        cvn_cfr_win_rate = cvn_cfr_wins / num_games * 100
        cvn_draw_rate = cvn_draws / num_games * 100

        cvn_nfsp_win_rates.append(cvn_nfsp_win_rate)
        cvn_cfr_win_rates.append(cvn_cfr_win_rate)
        cvn_draw_rates.append(cvn_draw_rate)

        elapsed_time = round(time.time() - start_time, 2)
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        log_message = f"""
NFSP vs CFRPlus results after {num_games*(iteration+1)} games:
NFSP wins: {nvc_nfsp_wins} ({nvc_nfsp_wins/num_games*100:.1f}%)
CFR+ wins: {nvc_cfr_wins} ({nvc_cfr_wins/num_games*100:.1f}%)
Draws: {nvc_draws} ({nvc_draws/num_games*100:.1f}%)
----------------------------------------------------------------
CFRPlus vs NFSP results after {num_games*(iteration+1)} games:
NFSP wins: {cvn_nfsp_wins} ({cvn_nfsp_wins/num_games*100:.1f}%)
CFR+ wins: {cvn_cfr_wins} ({cvn_cfr_wins/num_games*100:.1f}%)
Draws: {cvn_draws} ({cvn_draws/num_games*100:.1f}%)

Time elapsed: {minutes} mins and {seconds:02d} secs
"""
        write_to_log("/home/ethanso/CS175/comparisons/nfsp_vs_cfrplus_1", log_message) # run_6 fixes potential flipped player ids 
                                                                           # run 8 fixes the issue of using the wrong nfsp agent
                                                                            # run 10 is correctly loaded nfsp agents

    # After all iterations, create and save the graph
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    x = range(1, num_iterations + 1)
    plt.plot(x, nvc_nfsp_win_rates, label='NFSP')
    plt.plot(x, nvc_cfr_win_rates, label='CFR+')
    plt.plot(x, nvc_draw_rates, label='Draws')
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate (%)')
    plt.title('NFSP vs CFR+ Performance Over Iterations')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    x = range(1, num_iterations + 1)
    plt.plot(x, cvn_nfsp_win_rates, label='NFSP')
    plt.plot(x, cvn_cfr_win_rates, label='CFR+')
    plt.plot(x, cvn_draw_rates, label='Draws')
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate (%)')
    plt.title('CFR+ vs NFSP Performance Over Iterations')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the plot
    plt.savefig('/home/ethanso/CS175/comparisons/nfsp_vs_cfrplus_1/performance_graph.png')
    plt.close()













    # Clean up
    session.__exit__(None, None, None)


    # source venv/bin/activate
    # python3 nfsp_vs_cfrplus.py
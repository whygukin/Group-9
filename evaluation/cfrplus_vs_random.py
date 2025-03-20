import os
import matplotlib.pyplot as plt
import numpy as np
import time

import pyspiel
import gc
import pickle

from open_spiel.python import rl_environment
from open_spiel.python import policy
from open_spiel.python.algorithms import cfr
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import evaluate_bots
from open_spiel.python.bots import policy as bot_policy

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()  # Disable TF2 eager execution

class CustomPokerEnv(rl_environment.Environment):
    def __init__(self, params):
        self._game = pyspiel.load_game("universal_poker", params)
        super().__init__(self._game)

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
    session = tf.Session()
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


    cfrplus_save_dir = "/home/ethanso/CS175/cfrplus_1v1_models/run_3"

    cfr_solver = load_cfr_solver(cfrplus_save_dir, 500)

    session.run(tf.global_variables_initializer())

    # Create policy objects
    cfr_policy = cfr_solver.average_policy()
    random_policy = policy.UniformRandomPolicy(game)

    # Create policy bots
    rng = np.random.RandomState()
    cfr_bot_0 = bot_policy.PolicyBot(0, rng, cfr_policy)
    cfr_bot_1 = bot_policy.PolicyBot(1, rng, cfr_policy)

    random_bot_0 = bot_policy.PolicyBot(0, rng, random_policy)
    random_bot_1 = bot_policy.PolicyBot(1, rng, random_policy)

    cfr_random_bots = [cfr_bot_0, random_bot_1] 
    random_cfr_bots = [random_bot_0, cfr_bot_1]

    start_time = time.time()

    num_iterations = 10  # Number of iterations
    num_games = 1000  # Number of games per iteration
    

    rvc_random_win_rates = []
    rvc_cfr_win_rates = []
    rvc_draw_rates = []

    cvr_random_win_rates = []
    cvr_cfr_win_rates = []
    cvr_draw_rates = []


    for iteration in range(num_iterations):
        rvc_r_wins = 0
        rvc_cfr_wins = 0
        rvc_draws = 0

        cvr_r_wins = 0
        cvr_cfr_wins = 0
        cvr_draws = 0        
    
        all_returns_cfr_r = play_games(env.game, cfr_random_bots, num_games)
        all_returns_r_cfr = play_games(env.game, random_cfr_bots, num_games)

        # Count wins
        cvr_random_wins = sum(1 for r in all_returns_cfr_r if r[0] > r[1])
        cvr_cfr_wins = sum(1 for r in all_returns_cfr_r if r[1] > r[0])
        cvr_draws = sum(1 for r in all_returns_cfr_r if r[0] == r[1])

        rvc_random_wins = sum(1 for r in all_returns_r_cfr if r[0] > r[1])
        rvc_cfr_wins = sum(1 for r in all_returns_r_cfr if r[1] > r[0])
        rvc_draws = sum(1 for r in all_returns_r_cfr if r[0] == r[1])        

        # Calculate win rates
        cvr_random_win_rates.append(cvr_random_wins / num_games * 100)
        cvr_cfr_win_rates.append(cvr_cfr_wins / num_games * 100)
        cvr_draw_rates.append(cvr_draws / num_games * 100)

        rvc_random_win_rates.append(rvc_random_wins / num_games * 100)
        rvc_cfr_win_rates.append(rvc_cfr_wins / num_games * 100)
        rvc_draw_rates.append(rvc_draws / num_games * 100)

        elapsed_time = round(time.time() - start_time, 2)
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        log_message = f"""
Random vs CFR+ results after {num_games*(iteration+1)} games:
Random wins: {rvc_random_wins} ({rvc_random_wins/num_games*100:.1f}%)
CFR+ wins: {rvc_cfr_wins} ({rvc_cfr_wins/num_games*100:.1f}%)
Draws: {rvc_draws} ({rvc_draws/num_games*100:.1f}%)
----------------------------------------------------------------
CFR+ vs random results after {num_games*(iteration+1)} games:
random wins: {cvr_random_wins} ({cvr_random_wins/num_games*100:.1f}%)
CFR+ wins: {cvr_cfr_wins} ({cvr_cfr_wins/num_games*100:.1f}%)
Draws: {cvr_draws} ({cvr_draws/num_games*100:.1f}%)

Time elapsed: {minutes} mins and {seconds:02d} secs
"""
        write_to_log("/home/ethanso/CS175/comparisons/cfrplus_vs_random_1", log_message)

    # After all iterations, create and save the graph
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    x = range(1, num_iterations + 1)
    plt.plot(x, rvc_random_win_rates, label='random')
    plt.plot(x, rvc_cfr_win_rates, label='CFR+')
    plt.plot(x, rvc_draw_rates, label='Draws')
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate (%)')
    plt.title('random vs CFR+ Performance Over Iterations')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    x = range(1, num_iterations + 1)
    plt.plot(x, cvr_random_win_rates, label='random')
    plt.plot(x, cvr_cfr_win_rates, label='CFR+')
    plt.plot(x, cvr_draw_rates, label='Draws')
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate (%)')
    plt.title('CFR+ vs random Performance Over Iterations')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the plot
    plt.savefig('/home/ethanso/CS175/comparisons/cfrplus_vs_random_1/performance_graph.png')
    plt.close()

    # Clean up
    session.__exit__(None, None, None)


    # source venv/bin/activate
    # python3 cfrplus_vs_random.py
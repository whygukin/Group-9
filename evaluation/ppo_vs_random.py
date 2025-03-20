import os
import matplotlib.pyplot as plt
import numpy as np
import time

import pyspiel
import gc

from open_spiel.python import rl_environment
from open_spiel.python import policy
from open_spiel.python.bots import policy as bot_policy

from stable_baselines3 import PPO


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

class CustomPokerEnv(rl_environment.Environment):
    def __init__(self, params):
        self._game = pyspiel.load_game("universal_poker", params)
        super().__init__(self._game)



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

   
    ppo_loaded_model = PPO.load("ppo_poker_model_final")

    # Create policy bots
    rng = np.random.RandomState()
    ppo_bot_0 = bot_policy.PolicyBot(0, rng, PPOPolicy(env.game, player_id=0, ppo_model=ppo_loaded_model))
    ppo_bot_1 = bot_policy.PolicyBot(1, rng, PPOPolicy(env.game, player_id=1, ppo_model=ppo_loaded_model))

    
    random_policy = policy.UniformRandomPolicy(game)
    random_bot_0 = bot_policy.PolicyBot(0, rng, random_policy)
    random_bot_1 = bot_policy.PolicyBot(1, rng, random_policy)


    ppo_random_bots = [ppo_bot_0, random_bot_1]
    random_ppo_bots = [random_bot_0, ppo_bot_1]

    start_time = time.time()
    num_iterations = 10  # Number of iterations
    num_games = 1000  # Number of games per iteration

    pvc_ppo_win_rates = []
    pvc_random_win_rates = []
    pvc_draw_rates = []

    cvp_ppo_win_rates = []
    cvp_random_win_rates = []
    cvp_draw_rates = []


    for iteration in range(num_iterations):
        pvc_ppo_wins = 0
        pvc_random_wins = 0
        pvc_draws = 0

        cvp_ppo_wins = 0
        cvp_random_wins = 0
        cvp_draws = 0
    
        all_returns_PPO_random = play_games(env.game, ppo_random_bots, num_games)
        all_returns_random_PPO = play_games(env.game, random_ppo_bots, num_games)

        # Count wins
        pvc_ppo_wins = sum(1 for r in all_returns_PPO_random if r[0] > r[1])
        pvc_random_wins = sum(1 for r in all_returns_PPO_random if r[1] > r[0])
        pvc_draws = sum(1 for r in all_returns_PPO_random if r[0] == r[1])

        cvp_random_wins = sum(1 for r in all_returns_random_PPO if r[0] > r[1])
        cvp_ppo_wins = sum(1 for r in all_returns_random_PPO if r[1] > r[0])
        cvp_draws = sum(1 for r in all_returns_random_PPO if r[0] == r[1])

        # Calculate win rates
        pvc_ppo_win_rate = pvc_ppo_wins / num_games * 100
        pvc_random_win_rate = pvc_random_wins / num_games * 100
        pvc_draw_rate = pvc_draws / num_games * 100

        pvc_ppo_win_rates.append(pvc_ppo_win_rate)
        pvc_random_win_rates.append(pvc_random_win_rate)
        pvc_draw_rates.append(pvc_draw_rate)

        cvp_ppo_win_rate = cvp_ppo_wins / num_games * 100
        cvp_random_win_rate = cvp_random_wins / num_games * 100
        cvp_draw_rate = cvp_draws / num_games * 100

        cvp_ppo_win_rates.append(cvp_ppo_win_rate)
        cvp_random_win_rates.append(cvp_random_win_rate)
        cvp_draw_rates.append(cvp_draw_rate)

        elapsed_time = round(time.time() - start_time, 2)
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        log_message = f"""
----------------------------------------------------------------
PPO vs Random results after {num_games*(iteration+1)} games:
PPO wins: {pvc_ppo_wins} ({pvc_ppo_wins/num_games*100:.1f}%)
Random wins: {pvc_random_wins} ({pvc_random_wins/num_games*100:.1f}%)
Draws: {pvc_draws} ({pvc_draws/num_games*100:.1f}%)

Random vs PPO results after {num_games*(iteration+1)} games:
PPO wins: {cvp_ppo_wins} ({cvp_ppo_wins/num_games*100:.1f}%)
Random wins: {cvp_random_wins} ({cvp_random_wins/num_games*100:.1f}%)
Draws: {cvp_draws} ({cvp_draws/num_games*100:.1f}%)

Time elapsed: {minutes} mins and {seconds:02d} secs
----------------------------------------------------------------
"""
        write_to_log("/home/ethanso/CS175/comparisons/PPO_vs_random", log_message) # run_6 fixes potential flipped player ids 
                                                                           # run 8 fixes the issue of using the wrong nfsp agent

    # After all iterations, create and save the graph
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    x = range(1, num_iterations + 1)
    plt.plot(x, pvc_ppo_win_rates, label='PPO')
    plt.plot(x, pvc_random_win_rates, label='Random')
    plt.plot(x, pvc_draw_rates, label='Draws')
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate (%)')
    plt.title('PPO vs Random Performance Over Iterations')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    x = range(1, num_iterations + 1)
    plt.plot(x, cvp_ppo_win_rates, label='PPO')
    plt.plot(x, cvp_random_win_rates, label='Random')
    plt.plot(x, cvp_draw_rates, label='Draws')
    plt.xlabel('Iteration')
    plt.ylabel('Win Rate (%)')
    plt.title('Random vs PPO Performance Over Iterations')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save the plot
    plt.savefig('/home/ethanso/CS175/comparisons/PPO_vs_random/performance_graph.png')
    plt.close()











    # source venv/bin/activate
    # python3 ppo_vs_random.py
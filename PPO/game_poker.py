import gym
from gym import spaces
import numpy as np
import pyspiel
from stable_baselines3 import PPO
from treys import Evaluator, Card, Deck
import torch
import re

CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF = """\
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

# Hand evaluator
evaluator = Evaluator()

from treys import Deck, Evaluator, Card

evaluator = Evaluator()

def evaluate_hand_strength(hole_cards, board_cards, num_simulations=100):
    """
    Evaluates the hand strength of the player's hand.
    Uses Monte Carlo simulation to estimate hand strength.
    """
    win_count = 0

    # Simulate multiple games to estimate hand strength
    for _ in range(num_simulations):
        deck = Deck()
        remaining_deck_cards = [card for card in deck.cards if card not in (hole_cards + board_cards)]
        temp_deck = Deck()
        temp_deck.cards = remaining_deck_cards

        # Draw opponent's hand
        opponent_hand = temp_deck.draw(2)
        remaining_board = temp_deck.draw(5 - len(board_cards))

        final_board = board_cards + remaining_board
        my_score = evaluator.evaluate(final_board, hole_cards)
        opp_score = evaluator.evaluate(final_board, opponent_hand)

        if my_score < opp_score:
            win_count += 1
        elif my_score == opp_score:
            win_count += 0.5  # Tie condition

    # Return estimated win rate as hand strength
    return win_count / num_simulations


class PokerGymEnv(gym.Env):
    def __init__(self, learning_player_id=0):
        super().__init__()

        # Load the poker game using pyspiel (Universal Poker in this case)
        self.game = pyspiel.universal_poker.load_universal_poker_from_acpc_gamedef(CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF)

        # The player that the agent will control
        self.learning_player_id = learning_player_id
        self.opponent = None
        obs_dim = self.game.information_state_tensor_size()
        print("Game parameters:", self.game.get_parameters())
        print("Tensor state size:", obs_dim)

        act_dim = self.game.num_distinct_actions()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(act_dim)

        self.state = self.game.new_initial_state()

        print("Distinct actions in this game:", self.action_space, act_dim)

    def reset(self):
        """ Resets the environment to its initial state and returns the observation. """
        self.state = self.game.new_initial_state()
        obs = self.state.information_state_tensor(self.learning_player_id)
        return np.array(obs, dtype=np.float32).flatten()

    def step(self, action):
        """ Takes a single step in the environment. """
        done = False
        reward = 0
        info = {}

        current_player = self.state.current_player()

        while current_player != 0:
            # Random opponent action
            if current_player == 1 and self.opponent:
                legal_actions = self.state.legal_actions()
                action, _ = self.opponent.predict(np.array(self.state.information_state_tensor(1)), deterministic=False)
                if action not in legal_actions:
                    action = np.random.choice(legal_actions) if legal_actions else 0
                
                self.state.apply_action(action)
            else:
                legal_actions = self.state.legal_actions()
                opp_action = np.random.choice(legal_actions) if legal_actions else 0
                self.state.apply_action(opp_action)
            
            current_player = self.state.current_player()

            if self.state.is_terminal():
                done = True
                rewards = self.state.rewards()
                reward += rewards[self.learning_player_id]

                obs = self.state.information_state_tensor(self.learning_player_id)
                obs = np.array(obs, dtype=np.float32).flatten()

                return obs, reward, done, info

        # Agent's action
        if current_player == self.learning_player_id:
            
            obs_string = self.state.observation_string(self.learning_player_id)

            game_state = str(self.state)
            board_cards_start = game_state.find("BoardCards") + len("BoardCards")
            board_cards_str = game_state[board_cards_start:].split()[0]  
            board_cards = [board_cards_str[i:i+2] for i in range(0, len(board_cards_str), 2)]
            card_pattern = r"[2-9TJQKA][cdhs]"
            cards = re.findall(card_pattern, obs_string)
            
            legal_actions = self.state.legal_actions()

            if action not in legal_actions:
                #print(f"Doing illegal action {action}, legal_actions {legal_actions}")
                obs = self.state.information_state_tensor(self.learning_player_id)
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                obs_tensor = obs_tensor.unsqueeze(0) 
                distribution = model.policy.get_distribution(obs_tensor.to('cuda'))
                probs = distribution.distribution.probs
                probs_np = probs.cpu().detach().numpy()
                legal_mask = self.state.legal_actions_mask()
                masked_probs = probs_np * legal_mask
                action = np.argmax(masked_probs)
                reward -= 1 # illegal action.
            
            if board_cards_str != 'Node':
                hole_cards = [Card.new(card) for card in cards if isinstance(card, str)]  
                board_cards = [Card.new(card) for card in board_cards if isinstance(card, str)]  
                #print(f"HOLE CARDS: {hole_cards}, BOARD CARDS {board_cards}")
                hand_evaluation_score = evaluate_hand_strength(hole_cards, board_cards)
                #print(f"Hand Evaluation: {hand_evaluation_score}")
                if (action == 1 or action == 2) and hand_evaluation_score < 0.5:
                    reward -= (1 - hand_evaluation_score) # further discouragement
                elif (action == 1 or action == 2) and hand_evaluation_score > 0.5:
                    reward += (1 - hand_evaluation_score)
                elif (action == 0) and hand_evaluation_score < 0.5:
                    reward += (1 - hand_evaluation_score)
                elif (action == 0) and hand_evaluation_score > 0.5:
                    reward -= (1 - hand_evaluation_score)
        
            #print(f"Applied: {action} - {self.state.action_to_string(0, action)}!!!!")
            self.state.apply_action(action)

        if self.state.is_terminal():
            done = True
            rewards = self.state.rewards()
            reward += rewards[self.learning_player_id]


        obs = self.state.information_state_tensor(self.learning_player_id)
        obs = np.array(obs, dtype=np.float32).flatten()

        return obs, reward, done, info


    def render(self, mode="human"):
        """ Render the current state of the game (for debugging or visualization). """
        obs = self.state.information_state_tensor(self.learning_player_id)
        print("Current observation:", obs)

    def close(self):
        """ Close the environment and release any resources. """
        pass

    def add_opponent(self, opponent=None):
        self.opponent = opponent


if __name__ == "__main__":
    env = PokerGymEnv(learning_player_id=0)
    """

    model = PPO("MlpPolicy", env, verbose=1, gamma=0.99, ent_coef=0.15, tensorboard_log="./ppo_tensorboard/")
    model.learn(total_timesteps=1000000)  # Adjust the timesteps for training
    model.save("ppo_poker_model_final")


    
    env = PokerGymEnv(learning_player_id=0)  # Ensure PokerGymEnv is correctly implemented

    print("RAN")
    
    model = PPO("MlpPolicy", env, verbose=1, gamma=0.99, ent_coef=0.15, tensorboard_log="./ppo_tensorboard/")
    #model = PPO.load("ppo_poker_model")
    #model.set_env(env)

    #tmp_opp = PPO.load("ppo_poker_model")
    #tmp_opp.set_env(env)

    #env.add_opponent(tmp_opp)

    model.learn(total_timesteps=1000000)  # Adjust the timesteps for training
    model.save("ppo_poker_model_diff_v2_withTracker")
    """
    model = PPO.load("ppo_poker_model_final")
    state = env.reset()  # Reset the environment to start
    done = False
    numGames = 15000
    i = 0
    wins = 0
    actions = {}
    total_actions = 0

    while i < numGames:
        while not done:
            action, _ = model.predict(state)  # Get action from the trained model
            state, reward, done, _ = env.step(action)  # Take a step in the environment
            #print(f"Agent took action {action}")
            #print(f"Action: {action}, Reward: {reward}")

            if str(action) not in actions.keys():
                actions[str(action)] = 1
            else:
                actions[str(action)] += 1
            total_actions += 1

            if done:
                if reward > 0:
                    wins += 1
                env.reset() 
        
        i += 1
        done = False
        print(f"Playing Games {(i / numGames)*100:.2f}%", end='\r')
        

    print(f"Done! Total Wins {wins} out of {numGames}")
    print(actions)
    print("Total actions", total_actions)


    # List out all of the actions that the agent can do {Distribution of different actions that the agent takes}
    # Use stable_baselines3 to get the metrics (that are in each iteration and get the graphs for it.)
    # Test it on some agents and see how well it performs relative to the random agent
    # Test it against other well known-algorithms such as CFR / NFSP"


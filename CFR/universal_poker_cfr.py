#CITATION: CFR Algorithm from Google DeepMind's OpenSpiel, with parameters and game definition modified for Limit Texas Hold'em

import pickle
import sys
from absl import app
from absl import flags
import matplotlib.pyplot as plt
import pyspiel


universal_poker = pyspiel.universal_poker


FLAGS = flags.FLAGS


flags.DEFINE_enum("solver", "cfr", ["cfr", "cfrplus", "cfrbr"], "Which CFR solver to use")
_ITERATIONS = flags.DEFINE_integer("iterations", 100, "Number of iterations to run")


_CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF = """\
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




def simulate_performance_vs_random(game, solver_policy, random_policy, num_episodes=1000):
   """Simulates solver_policy (Player 0) vs random_policy (Player 1) for num_episodes.


   Args:
       game: A pyspiel.Game object.
       solver_policy: The policy (CFR average) for Player 0.
       random_policy: A random (or any other) policy for Player 1.
       num_episodes: How many episodes to simulate.


   Returns:
       (avg_return_p0, win_rate_p0):
           avg_return_p0: Average return for Player 0 across episodes.
           win_rate_p0: Fraction of episodes where Player 0's final return > Player 1's final return.
   """
   import random


   total_return_p0 = 0.0
   wins_for_p0 = 0


   for _ in range(num_episodes):
       state = game.new_initial_state()
       while not state.is_terminal():
           if state.is_chance_node():
               # Sample chance outcomes (e.g. dealing cards)
               outcomes, probs = zip(*state.chance_outcomes())
               action = random.choices(outcomes, weights=probs, k=1)[0]
               state.apply_action(action)
           else:
               current_player = state.current_player()
               if current_player == 0:
                   # Use solver's average policy for Player 0
                   action_prob_dict = solver_policy.action_probabilities(state)
               else:
                   # Use random policy for Player 1
                   action_prob_dict = random_policy.action_probabilities(state)


               actions = state.legal_actions()
               action_probs = [action_prob_dict.get(a, 0.0) for a in actions]


               # Sample an action from those probabilities
               action = random.choices(actions, weights=action_probs, k=1)[0]
               state.apply_action(action)


       # Once terminal, get final returns
       returns = state.returns()
       total_return_p0 += returns[0]
       # Count a "win" if P0's return > P1's return
       if returns[0] > returns[1]:
           wins_for_p0 += 1


   avg_return_p0 = total_return_p0 / num_episodes
   win_rate_p0 = wins_for_p0 / num_episodes
   return avg_return_p0, win_rate_p0




def main(_):
   # 1. Load the custom game from ACPC definition
   game = universal_poker.load_universal_poker_from_acpc_gamedef(
       _CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF
   )


   # 2. Initialize the chosen solver
   if FLAGS.solver == "cfr":
       solver = pyspiel.CFRSolver(game)
   elif FLAGS.solver == "cfrplus":
       solver = pyspiel.CFRPlusSolver(game)
   elif FLAGS.solver == "cfrbr":
       solver = pyspiel.CFRBRSolver(game)
   else:
       print("Unknown solver:", FLAGS.solver)
       sys.exit(1)


   # Create a random policy (uniform over legal actions).
   random_policy = pyspiel.UniformRandomPolicy(game)


   # Arrays to store metrics
   exploitabilities = []
   performance_vs_random = []  # average return vs random
   win_rates_vs_random = []    # fraction of episodes P0 "wins"


   # 3. Main training loop
   for i in range(_ITERATIONS.value):
       # Perform one CFR update
       solver.evaluate_and_update_policy()


       # Compute exploitability
       current_exploit = pyspiel.exploitability(game, solver.average_policy())
       exploitabilities.append(current_exploit)


       # Measure performance vs random
       avg_return_p0, p0_win_rate = simulate_performance_vs_random(
           game, solver.average_policy(), random_policy, num_episodes=500
       )
       performance_vs_random.append(avg_return_p0)
       win_rates_vs_random.append(p0_win_rate)


       # Print log
       print(f"Iteration {i} | Exploitability: {current_exploit:.6f} | "
             f"AvgReturn vs Random: {avg_return_p0:.3f} | Win Rate: {p0_win_rate:.3f}")


       # Optionally: every 20 iterations, do an extra policy update
       if i > 0 and i % 20 == 0:
           print(f"==> Iteration {i}: Extra 'update' after random match check.")
           solver.evaluate_and_update_policy()


   # 4. Save the solver state to a file
   filename = f"/tmp/{FLAGS.solver}_solver.pickle"
   print("Persisting the model to:", filename)
   with open(filename, "wb") as file:
       pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)


   # 5. Load the solver from the file
   print("Loading the model from:", filename)
   with open(filename, "rb") as file:
       loaded_solver = pickle.load(file)


   # 6. Check exploitability of the loaded solver
   loaded_exploit = pyspiel.exploitability(game, loaded_solver.average_policy())
   print(f"Exploitability of the loaded model: {loaded_exploit:.6f}")


   # 7. Final performance vs random
   final_avg_return, final_win_rate = simulate_performance_vs_random(
       game, loaded_solver.average_policy(), random_policy, num_episodes=1000
   )
   print(f"Final performance vs random -> AvgReturn: {final_avg_return:.3f}, "
         f"Win Rate: {final_win_rate:.3f}")


   #
   # 8. Plot and save Exploitability vs. Iteration
   #
   plt.figure(figsize=(8, 5))
   plt.plot(range(len(exploitabilities)), exploitabilities, marker='o')
   plt.title("CFR Training: Exploitability vs. Iteration")
   plt.xlabel("Iteration")
   plt.ylabel("Exploitability")
   plt.grid(True)
   plt.savefig("exploitability_plot.png")  # save to disk
   plt.show()
   print("Exploitability plot saved to 'exploitability_plot.png'")


   #
   # 9. Plot and save Average Return vs. Iteration
   #
   plt.figure(figsize=(8, 5))
   plt.plot(range(len(performance_vs_random)), performance_vs_random, marker='o')
   plt.title("CFR Training: Average Return vs. Random Agent")
   plt.xlabel("Iteration")
   plt.ylabel("Average Return (Player 0)")
   plt.grid(True)
   plt.savefig("avg_return_plot.png")  # save to disk
   plt.show()
   print("Average Return plot saved to 'avg_return_plot.png'")


   #
   # 10. Plot and save Win Rate vs. Iteration
   #
   plt.figure(figsize=(8, 5))
   plt.plot(range(len(win_rates_vs_random)), win_rates_vs_random, marker='o')
   plt.title("CFR Training: Win Rate vs. Random Agent")
   plt.xlabel("Iteration")
   plt.ylabel("Win Rate (Player 0)")
   plt.grid(True)
   plt.savefig("win_rate_plot.png")  # save to disk
   plt.show()
   print("Win Rate plot saved to 'win_rate_plot.png'")




if __name__ == "__main__":
   app.run(main)

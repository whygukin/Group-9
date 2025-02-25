#CITATION: CFR Algorithm from Google DeepMind's OpenSpiel, with parameters and gamedef modified for Texas Hold'em


import pickle
import sys
from absl import app
from absl import flags
import matplotlib.pyplot as plt   # <-- For plotting


import pyspiel


universal_poker = pyspiel.universal_poker


FLAGS = flags.FLAGS


flags.DEFINE_enum("solver", "cfr", ["cfr", "cfrplus", "cfrbr"], "CFR solver")
_ITERATIONS = flags.DEFINE_integer("iterations", 100, "Number of iterations")


CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF = """\
GAMEDEF
limit
numPlayers = 2
numRounds = 1
blind = 2 4
raiseSize = 4 4 8
firstPlayer = 1
maxRaises = 2 2 2
numSuits = 2
numRanks = 5
numHoleCards = 1
numBoardCards = 0 2 1
stack = 20
END GAMEDEF
"""


def main(_):
 # 1. Load the custom game from an ACPC definition.
 game = universal_poker.load_universal_poker_from_acpc_gamedef(
     CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF
 )


 # 2. Initialize the chosen solver (CFR, CFR+, or CFR-BR).
 if FLAGS.solver == "cfr":
   solver = pyspiel.CFRSolver(game)
 elif FLAGS.solver == "cfrplus":
   solver = pyspiel.CFRPlusSolver(game)
 elif FLAGS.solver == "cfrbr":
   solver = pyspiel.CFRBRSolver(game)
 else:
   print("Unknown solver")
   sys.exit(0)


 # This list will track exploitability at each iteration
 exploitabilities = []


 # 3. Main training loop (first half of the iterations).
 #    Evaluate and update the policy at each iteration, record exploitability.
 for i in range(int(_ITERATIONS.value / 2)):
   solver.evaluate_and_update_policy()
   current_exploit = pyspiel.exploitability(game, solver.average_policy())
   exploitabilities.append(current_exploit)
   print(f"Iteration {i} exploitability: {current_exploit:.6f}")


 # 4. Save the solver state to a file
 filename = f"/tmp/{FLAGS.solver}_solver.pickle"
 print("Persisting the model...")
 with open(filename, "wb") as file:
   pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)


 # 5. Load the solver from the file
 print("Loading the model...")
 with open(filename, "rb") as file:
   loaded_solver = pickle.load(file)


 # 6. Check exploitability of the loaded solver
 loaded_exploit = pyspiel.exploitability(game, loaded_solver.average_policy())
 print(f"Exploitability of the loaded model: {loaded_exploit:.6f}")


 # Continue training (second half of the iterations) from the loaded solver
 for i in range(int(_ITERATIONS.value / 2)):
   loaded_solver.evaluate_and_update_policy()
   current_exploit = pyspiel.exploitability(game, loaded_solver.average_policy())
   exploitabilities.append(current_exploit)
   iteration_label = int(_ITERATIONS.value / 2) + i
   print(f"Iteration {iteration_label} exploitability: {current_exploit:.6f}")


 # 7. Plot the exploitability over all iterations
 plt.figure(figsize=(8, 5))
 plt.plot(range(len(exploitabilities)), exploitabilities, marker='o')
 plt.title("CFR Training: Exploitability vs. Iteration")
 plt.xlabel("Iteration")
 plt.ylabel("Exploitability")
 plt.grid(True)
  # Save the plot to a file (or use plt.show() if running interactively).
 plt.savefig("exploitability_plot3.png")
 plt.show()
 print("Exploitability plot saved to /tmp/exploitability_plot2.png")


if __name__ == "__main__":
 app.run(main)
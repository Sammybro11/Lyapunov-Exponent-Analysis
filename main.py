
import numpy as np
from Lyapunov import Lyapunov_Exponent_Henon
from graphing import Henon_Graph, Henon_Simulation
from Model_Structures import Create_Henon_Map
from tqdm import tqdm

a_vals = np.linspace(1.2, 1.5, 61)
b_vals = np.linspace(0.15, 0.35, 41)

results = []

for a in tqdm(a_vals):
    for b in b_vals:
        lyap = Lyapunov_Exponent_Henon(a, b)
        results.append({"a": np.round(a,3), "b": np.round(b,3), "l1": lyap[0], "l2": lyap[1]})

Henon_Graph(results)

# Creating Animations for different starting conditions

a = 1.5
b = 0.35
Henon_Func = Create_Henon_Map(a, b)

Henon_Sim = Henon_Simulation(Henon_Func, 30, 100, "Divergent")
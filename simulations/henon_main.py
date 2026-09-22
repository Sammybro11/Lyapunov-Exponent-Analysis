import numpy as np
from src.Lyapunov import Batched_Lyapunov_Exponent_Henon
from utils.graphing import Henon_Graph, Map_Simulation
from models.Model_Structures import Create_Henon_Map

a_vals = np.linspace(1.2, 1.5, 61)
b_vals = np.linspace(0.18, 0.38, 41)

# Batched Method
results = Batched_Lyapunov_Exponent_Henon(a_vals, b_vals, batch_size = 4096)

Henon_Graph(results)

# Creating Animations for different starting conditions

a = 1.5
b = 0.35
Henon_Func = Create_Henon_Map(a, b)

Henon_Sim = Map_Simulation(Henon_Func, 40, 500, "Divergent", "Divergent (a = 1.5 b = 0.35)")

a = 1.4
b = 0.3
Henon_Func = Create_Henon_Map(a, b)

Henon_Sim = Map_Simulation(Henon_Func, 40, 500, "Attractor", "Chaotic Attractor (a = 1.4 b = 0.3)")

a = 1.44
b = 0.215
Henon_Func = Create_Henon_Map(a, b)

Henon_Sim = Map_Simulation(Henon_Func, 40, 500, "Convergent", "Convergent (a = 1.44 b = 0.215)")

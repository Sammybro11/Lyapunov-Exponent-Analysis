from src.Lyapunov import Lyapunov_Exponent_Peter
from utils.graphing import Peter_Attractor, Map_Simulation
from models.Model_Structures import Create_Peter_Map

# Classic chaotic Peter de Jong parameters
a, b, c, d = 1.4, -2.3, 2.4, -2.1

l1, l2 = Lyapunov_Exponent_Peter(a, b, c, d)
print(f"Peter de Jong (a={a}, b={b}, c={c}, d={d}): lambda1 = {l1:.4f}, lambda2 = {l2:.4f}")

Peter_Func = Create_Peter_Map(a, b, c, d)

Peter_Attractor(Peter_Func, 100000, "peter_attractor", f"Peter de Jong Attractor (a={a} b={b} c={c} d={d})")

Map_Simulation(Peter_Func, 40, 500, "Peter_Attractor", f"Peter de Jong (a={a} b={b} c={c} d={d})", bounds=(-2.5, 2.5, -2.5, 2.5))

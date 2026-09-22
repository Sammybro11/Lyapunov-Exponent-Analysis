import numpy as np
from models.Model_Structures import Create_Henon_Map
import torch

EPS = 1e-10

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def Batched_Lyapunov_Exponent_Henon(a_vals, b_vals, batch_size: int):
    device = get_device()
    transient = 3000
    n_iters = 5000
    results = []

    A, B = np.meshgrid(a_vals, b_vals, indexing = "ij")
    A_flat = A.ravel()
    B_flat = B.ravel()

    N = len(A_flat)
    Map = Create_Henon_Map(0.0, 0.0, batched = True)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)

        # loading batches
        a_batch = torch.tensor(A_flat[start:end], dtype = torch.float32, device = device)
        b_batch = torch.tensor(B_flat[start:end], dtype = torch.float32, device = device)

        batch_n = a_batch.shape[0]

        # init points at [0.2 , 0.2]
        coords_batch = torch.full((batch_n, 2), 0.2, dtype = torch.float32, device = device)


        sum_log = torch.zeros((batch_n, 2), dtype = torch.float32, device = device)

        for _ in range(transient):
            coords_batch = Map(coords_batch, a_batch, b_batch)

        Q = torch.eye(2, dtype = torch.float32, device = device).expand(batch_n, 2, 2).clone()

        for _ in range(n_iters):
            coords_batch = Map(coords_batch, a_batch, b_batch)

            Jacob = torch.stack([
                torch.stack([
                    -2.0 * a_batch * coords_batch[:, 0], torch.ones_like(coords_batch[:, 0])],dim = 1),
                torch.stack([
                    b_batch, torch.zeros_like(coords_batch[:, 0])], dim = 1)],
                dim = 1)

            P = torch.matmul(Jacob, Q)
            Q, R = torch.linalg.qr(P)

            diag = torch.abs(torch.diagonal(R, 0, -2, -1))
            diag = torch.clamp(diag, min = EPS)
            sum_log += torch.log(diag)

        lyap = (sum_log / n_iters ).cpu().numpy()

        for i in range(batch_n):
            results.append({
                "a": float(A_flat[start + i]),
                "b": float(B_flat[start + i]),
                "l1": float(lyap[i, 0]),
                "l2": float(lyap[i, 1])
                })

    return results


def Lyapunov_Exponent_Henon(a, b):
    Map = Create_Henon_Map(a,b, False)

    init_coordinates = np.array([0.2, 0.2])

    perturbation_vectors = np.eye(2, dtype=float)

    sum_log = np.zeros(2, dtype = float)
    transient = 3000
    n_iterations = 5000
    coords = init_coordinates

    for _ in range(transient):
        coords = Map(coords)
        if np.any(np.isnan(coords)):
            return np.array([np.nan, np.nan])

    for _ in range(n_iterations):
        coords = Map(coords)
        if np.any(np.isnan(coords)):
            return np.array([np.nan, np.nan])

        Jacobian = np.array([[-2 * a * coords[0], 1],
                    [b, 0]], dtype=float)
        perturbation = Jacobian @ perturbation_vectors

        Q, R = np.linalg.qr(perturbation)

        diag = np.maximum(np.abs(np.diag(R)), EPS) # Norm of each vector before normalization, the "stretch"
        sum_log += np.log(diag)

        perturbation_vectors = Q

    return sum_log/n_iterations

if __name__ == "__main__":
    print("Henon (a=1.4, b=0.3):", Lyapunov_Exponent_Henon(1.4, 0.3))

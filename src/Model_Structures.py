import numpy as np
import torch

# Discrete Maps

def Create_Henon_Map(a, b, batched = False):
    # Creates a discrete function of a Henon Map for a given parameters

    def Henon_Map(coords):
        if abs(coords[0]) > 1e100 or abs(coords[1]) > 1e100:
            return np.array([np.nan, np.nan])
        x_next = 1 - a * np.power(coords[0], 2) + coords[1]
        y_next = b*coords[0]

        return np.array([x_next, y_next])

    def Batched_Map(coords, a_batch, b_batch):
        x = coords[:, 0]
        y = coords[:, 1]
        x_next = 1.0 - a_batch * x * x + y
        y_next = b_batch * y
        return torch.stack([x_next, y_next], dim = 1)

    if batched:
        return Batched_Map
    else:
        return Henon_Map

def Create_Peter_Map(a, b, c, d):

    def Peter_Jong(coords):
        x_next = np.sin(a * coords[1]) - np.cos(b*coords[0])
        y_next = np.sin(c * coords[1]) - np.cos(d*coords[0])

        return np.array([x_next, y_next])
    return Peter_Jong

gravity = 9.80665

def DoublePendulum(state: np.ndarray, time: float, struct: np.ndarray) -> np.ndarray:
    """

    :param state: [ Theta_1, Theta_2 , Omega_1, Omega_2 ]
    :param time: Current time
    :param struct: [mass_1, mass_2, length_1, length_2]
    :return: derivative_state = [ Omega_1, Omega_2, Accel_1, Accel_2 ]

    Equations:

    """

    dy_dx = np.zeros_like(state)
    dy_dx[0] = state[2]
    dy_dx[1] = state[3]
    
    mass_1, mass_2, length_1, length_2 = struct
    theta_1, theta_2, omega_1, omega_2 = state

    diff_theta = theta_1 - theta_2

    numerator_1 = -(
        (gravity * (2 * mass_1 + mass_2) * np.sin(theta_1)) # g(2m_1 + m_2)sin(0_1)
        + (mass_2 * gravity * np.cos(diff_theta - theta_2)) # gm_2 cos(0_1 - 2*0_2)
        + (2 * np.sin(diff_theta) * mass_2 * ((omega_2**2) * length_2 + (omega_1**2) * length_1 * np.cos(diff_theta)))
        # (2 sin(0_1 - 0_2) m_2 ( w_2^2 L_2 + w_1^2 L_1 cos(0_1 - 0_2) )
    )

    denominator_1 = length_1 * (2 * mass_1 + mass_2 - mass_2 * np.cos(2* diff_theta))
    dy_dx[2] = numerator_1 / denominator_1

    numerator_2 = 2 * np.sin(diff_theta) * (
        (omega_1**2 * length_1 * (mass_1 + mass_2) )
        + (gravity * (mass_1 + mass_2) * np.cos(theta_1))
        + (omega_2**2 * length_2 * mass_2 * np.cos(diff_theta) )
    )
    denominator_2 = length_2 * (2 * mass_1 + mass_2 - mass_2 * np.cos(2* diff_theta))

    dy_dx[3] = numerator_2 / denominator_2

    return dy_dx









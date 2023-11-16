"""
 ComputeStageCosts.py

 Python function template to compute the stage cost matrix.

 Dynamic Programming and Optimal Control
 Fall 2023
 Programming Exercise
 
 Contact: Antonio Terpin aterpin@ethz.ch
 
 Authors: Abhiram Shenoi, Philip Pawlowsky, Antonio Terpin

 --
 ETH Zurich
 Institute for Dynamic Systems and Control
 --
"""

import numpy as np


def compute_stage_cost(Constants):
    """Computes the stage cost matrix for the given problem.

    It is of size (K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - G[i,l] corresponds to the cost incurred when using input l
            at state i.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Stage cost matrix G of shape (K,L)
    """
    K = Constants.T * Constants.D * Constants.N * Constants.M
    input_space = np.array([Constants.V_DOWN, Constants.V_STAY, Constants.V_UP])
    L = len(input_space)

    # Position of the Sun:
    def x_sun(t): return np.floor((Constants.M - 1) * (Constants.T - 1 - t)/(Constants.T - 1))

    # Solar Energy Cost:
    def g_solar_computation():
        solar_cost = np.empty((Constants.T, Constants.M)) # Initialize solar_cost matrix T x M
        c = np.array([-1, 0, 1]) # Constants for the minimization according to eq. (4)
        min_val = np.empty(c.size)
        # Now we fill up the matrix for each Time and possible x Value of the balloon!
        for t in range(Constants.T): # Loop over time t
            pos_sun = x_sun(t) # position of the sun at time t
            for x in range(Constants.M):
                for i_c in range(c.size): min_val[i_c] = ((x + Constants.M * c[i_c]) - pos_sun) ** 2 
                solar_cost[t,x] = np.min(min_val)
        return solar_cost
    
    def g_connectivity_computation():
        con_cost = np.empty((Constants.M, Constants.N)) # Matrix with size M x N
        c = np.array([-1, 0, 1]) # Constants for the minimization according to eq. (4)
        min_val = np.empty(c.size)

        for x in range(Constants.M):
            for y in range(Constants.N):
                acc_cost = 0
                for city in Constants.CITIES_LOCATIONS:
                    for i_c in range(c.size): 
                        # Note: Lambda * z is missing but since this is a linear process it can be multiplied afterwards!
                        min_val[i_c] = np.sqrt((x + Constants.M * c[i_c] - city[1]) ** 2 + (y - city[0]) ** 2)
                    
                    acc_cost += np.min(min_val)
                con_cost[x,y] = acc_cost
        return con_cost


    g_city = g_connectivity_computation()
    g_solar = g_solar_computation()

    G = np.ones((K, L)) * np.inf

    # Again step sizes need to be defined:
    x_step = 1
    y_step = Constants.M
    z_step = Constants.M * Constants.N
    t_step = Constants.M * Constants.N * Constants.D

    # Discount Factor?

    for t in range(Constants.T):
        for z in range(Constants.D):
            for y in range(Constants.N):
                for x in range(Constants.M):
                    pos = x * x_step + y * y_step + z * z_step + t * t_step # Current Position of the Balloon

                    # First, handling of the special cases:
                    # Boundary 1 at the bottom (z == 0)
                    if z == 0:
                        # Not allowed to go DOWN! Only UP or STAY
                        G[pos, Constants.V_UP] = G[pos, Constants.V_STAY] = \
                            (g_city[x,y] + Constants.LAMBDA_LEVEL * z) + \
                            Constants.LAMBDA_TIMEZONE * g_solar[t,x]
                        
                    # Boundary 2 at the top (z == D - 1)
                    elif z == (Constants.D - 1):
                        # Not allowed to go UP! Only DOWN or STAY
                        G[pos, Constants.V_DOWN] = G[pos, Constants.V_STAY] = \
                            (g_city[x,y] + Constants.LAMBDA_LEVEL * z) + \
                            Constants.LAMBDA_TIMEZONE * g_solar[t,x]
                    
                    # Handle the general case:
                    else:
                        G[pos, input_space] = (g_city[x,y] + Constants.LAMBDA_LEVEL * z) + \
                            Constants.LAMBDA_TIMEZONE * g_solar[t,x]

    return G

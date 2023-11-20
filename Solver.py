"""
 Solver.py

 Python function template to solve the discounted stochastic
 shortest path problem.

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


def solution(P, G, alpha):
    """Computes the optimal cost and the optimal control input for each 
    state of the state space solving the discounted stochastic shortest
    path problem by:
            - Value Iteration;
            - Policy Iteration;
            - Linear Programming; 
            - or a combination of these.

    Args:
        P  (np.array): A (K x K x L)-matrix containing the transition probabilities
                       between all states in the state space for all control inputs.
                       The entry P(i, j, l) represents the transition probability
                       from state i to state j if control input l is applied
        G  (np.array): A (K x L)-matrix containing the stage costs of all states in
                       the state space for all control inputs. The entry G(i, l)
                       represents the cost if we are in state i and apply control
                       input l
        alpha (float): The discount factor for the problem

    Returns:
        np.array: The optimal cost to go for the discounted stochastic SPP
        np.array: The optimal control policy for the discounted stochastic SPP

    """

    K = G.shape[0]

    J_opt = np.zeros(K)
    u_opt = np.zeros(K)
    
    # TODO implement Value Iteration, Policy Iteration, 
    #      Linear Programming or a combination of these

    # 1) Value Iteration:

    # Initialize for each state with zero initially!
    # Note number of states is K, where K = T * M * N * D
    # and L is the number of inputs, where L = [0, 1, 2], DOWN, UP, STAY
    
    eps = 1e-6
    max_iterations = 100 # can be changed for instance until convergence is reached?

    # Inputs:
    L = np.array([0, 1, 2]) 
    # 0: DOWN
    # 1: STAY
    # 2: UP

    for _ in range(max_iterations):
        J_updated = np.zeros(K) # Update (Initially all zeros!)
        for i in range(K): # Iteration through every State!
            Q = np.zeros(L.size)
            for input in range(L.size):
                Q[input] = G[i, input] + alpha * np.sum(P[i, :, input] * J_opt)
            J_updated[i] = np.min(Q) 
            u_opt[i] = np.argmin(Q) # returns the index of where Q is min!

        if np.max(np.abs(J_updated - J_opt)) < eps:
            break

        J_opt = J_updated.copy()

    return J_opt, u_opt

def freestyle_solution(Constants):
    """Computes the optimal cost and the optimal control input for each 
    state of the state space solving the discounted stochastic shortest
    path problem with a 200 MiB memory cap.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: The optimal cost to go for the discounted stochastic SPP
        np.array: The optimal control policy for the discounted stochastic SPP
    """
    K = Constants.T * Constants.D * Constants.N * Constants.M

    J_opt = np.zeros(K)
    u_opt = np.zeros(K)
    
    # TODO implement a solution that not necessarily adheres to
    #      the solution template. You are free to use
    #      compute_transition_probabilities and
    #      compute_stage_cost, but you are also free to introduce
    #      optimizations.

    return J_opt, u_opt

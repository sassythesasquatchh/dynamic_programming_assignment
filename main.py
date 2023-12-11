"""
 main.py

 Python script that calls all the functions for computing the optimal cost
 and policy of the given problem.

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
import itertools
import sys

from Constants import Constants
from ComputeTransitionProbabilities import compute_transition_probabilities
from ComputeStageCosts import compute_stage_cost
from Solver import solution, freestyle_solution

if __name__ == "__main__":
    print("Generating state space and input space...")
    # State space
    # np.arange returns an ndarray of evenly spaced values between start and stop
    t = np.arange(0, Constants.T)
    z = np.arange(0, Constants.D)
    y = np.arange(0, Constants.N)
    x = np.arange(0, Constants.M)

    # returns an ndarray of all of the ordered pairs of the sets containing the possible states in each dimension, ie. the cartesion product of x,y,z,t
    state_space = np.array(list(itertools.product(t, z, y, x)))
    K = len(state_space)

    # input space
    input_space = np.array([Constants.V_DOWN, Constants.V_STAY, Constants.V_UP])
    L = len(input_space)

    # Set the following to True as you progress with the files
    transition_probabilities_implemented = True
    stage_costs_implemented = True
    solution_implemented = True
    freestyle_solution_implemented = False

    # Compute transition probabilities
    if transition_probabilities_implemented and not freestyle_solution_implemented:
        sys.stdout.write("[ ] Computing transition probabilities...")

        """
            Compute the transition probabilities between all states in the
            state space for all control inputs.
            The transition probability matrix has the dimension (K x K x L), i.e.
            the entry P(i, j, l) represents the transition probability from state i
            to state j if control input l is applied.
        """

        # TODO implement this function in ComputeTransitionProbabilities.py
        P = compute_transition_probabilities(Constants)
        print("\r[X] Transition probabilities computed.     ")
    else:
        print(
            "[ ] Transition probabilities not implemented. If this is unexpected, check the boolean 'transition_probabilities_implemented'."
        )
        P = np.zeros((K, K, L))

    # Compute stage costs
    if stage_costs_implemented and not freestyle_solution_implemented:
        sys.stdout.write("[ ] Computing stage costs...")

        """
            Compute the stage costs for all states in the state space for all
            control inputs.
            The stage cost matrix has the dimension (K x L), i.e. the entry G(i, l)
            represents the cost if we are in state i and apply control input l.
        """

        # TODO implement this function in ComputeStageCosts.py
        G = compute_stage_cost(Constants)

        print("\r[X] Stage costs computed.            ")
    else:
        print(
            "[ ] Stage costs not implemented. If this is unexpected, check the boolean 'stage_costs_implemented'."
        )
        G = np.ones((K, L)) * np.inf

    # Solve the stochastic shortest path problem
    if solution_implemented and not freestyle_solution_implemented:
        import tracemalloc
        import time

        sys.stdout.write("[ ] Solving discounted stochastic shortest path problem...")
        tracemalloc.start()
        start_time = time.time()

        # TODO implement this function in Solver.py
        J_opt, u_opt = solution(P, G, Constants.ALPHA)
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        #############################################
        state_space_size = G.shape[0]
        print("State Space Size: ", state_space_size)
        #############################################

        print("\r[X] Discounted stochastic shortest path problem solved.    ")
        print("Elapsed time: {:.6f} seconds".format(end_time - start_time))
        print("Peak memory usage in MiB: {:.4}".format(peak / 2**20))
    else:
        print(
            "[ ] Solution of the discounted stochastic shortest path problem not implemented. If this is unexpected, check the boolean 'solution_implemented'."
        )
        J_opt = np.inf * np.ones(K)
        u_opt = np.zeros(K)

    if freestyle_solution_implemented:
        import tracemalloc
        import time
        # First step we want to compute P and G!
        P = compute_transition_probabilities(Constants)
        G = compute_stage_cost(Constants)

        sys.stdout.write("[ ] Solving discounted stochastic shortest path problem...")
        start_time = time.time()
        tracemalloc.start()

        # TODO implement this function in Solver.py
        J_opt, u_opt = freestyle_solution(Constants, P, G)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        print("\r[X] Discounted stochastic shortest path problem solved.   ")
        print("Peak memory usage in MiB: {:.4}".format(peak / 2**20))
        print("Elapsed time: {:.6f} seconds".format(end_time - start_time))
    else:
        print(
            "[ ] Freestyle solution not implemented. If this is unexpected, check the boolean 'freestyle_solution_implemented'."
        )

    # Do not change the variable names
    # Saving the workspace so it can be accessed for vizualization
    sys.stdout.write("Saving workspace...")
    np.savez(
        "./workspaces/workspace_",
        P=P,
        G=G,
        J=J_opt,
        u=u_opt,
        T=Constants.T,
        D=Constants.D,
        N=Constants.N,
        M=Constants.M,
        LOC_CITIES=Constants.CITIES_LOCATIONS,
    )

    # Terminated
    print("\rWorkspace saved for inspection.    ")

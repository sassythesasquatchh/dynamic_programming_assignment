"""
 ComputeTransitionProbabilities.py

 Python function template to compute the transition probability matrix.

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


def compute_transition_probabilities(Constants):
    """Computes the transition probability matrix P.

    It is of size (K,K,L) where:
        - K is the size of the state space;
        - L is the size of the input space; and
        - P[i,j,l] corresponds to the probability of transitioning
            from the state i to the state j when input l is applied.

    Args:
        Constants: The constants describing the problem instance.

    Returns:
        np.array: Transition probability matrix of shape (K,K,L).
    """
    K = Constants.T * Constants.D * Constants.N * Constants.M
    input_space = np.array([Constants.V_DOWN, Constants.V_STAY, Constants.V_UP])
    L = len(input_space)

    P = np.zeros((K, K, L))

    # TODO fill the transition probability matrix P here

    # For each Input we have the compute the probability! Possible inputs are:
    # UP, DOWN and STAY!

    # Define step sizes:
    x_step = 1
    y_step = Constants.M
    z_step = Constants.M * Constants.N
    t_step = Constants.M * Constants.N * Constants.D

    t_shift = Constants.T * t_step # Not sure here!!!

    # These were derived due to the given dynamics:
    # For the Time:
    # t_k1 = t_k0 + 1; thus the stepsize is t = 1!
    # Note: Same for x_k1 and y_k1!

    # All together, the Position can be evaluated by:

    # One step on the x direction:
    # pos(t_k1,x_k1,y_k0,z_k0) = pos(t_k0,x_k0,y_k0,z_k0) + t_step + x_step

    # One step on the x and y direction:
    # pos(t_k1,x_k1,y_k1,z_k0) = pos(t_k0,x_k0,y_k0,z_k0) + t_step + x_step - y_step

    # Note: This function can be used to write less code!
    def t_transform(t, Constants, t_shift):
        if t != (Constants.T - 1):
            return 0
        else:
            return t_shift
        
    def u_boundary(u, Constants):
        # Depending on u the boundary changes!
        if u == Constants.V_DOWN:
            return 0 # For Down z = 0 is not allowed!
        elif u == Constants.V_UP:
            return (Constants.D - 1)
        elif u == Constants.V_STAY:
            return np.nan # For Stay it doesn't matter!
        
    def move(u, Constants, z_step):
        if u == Constants.V_DOWN:
            return -z_step
        elif u == Constants.V_UP:
            return z_step


    # # 1) Input DOWN!
    # for t in range(Constants.T): # Time
    #     for z in range(Constants.D): # z-Direction (Atmospheric level)
    #         for y in range(Constants.N): # y-Direction (Note: Boundary at y = 0 and y = N-1)
    #             for x in range(Constants.M): # x-Direction (Note: Periodic!)
    #                 # Position:
    #                 pos = t * t_step + x * x_step + y * y_step + z * z_step

    #                 # Calculate the Probability of actually moving from i to j!

    #                 if t != (Constants.T - 1): # The time is < T-1
    #                     if z != 0: # Also I am not at the boundary z == 0!
    #                         # STAY in Z:
    #                         # STAY (Don't move at all!)
    #                         P[pos, pos + t_step, Constants.V_DOWN] = Constants.ALPHA \
    #                             * Constants.P_V_TRANSITION[0] \
    #                             * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_STAY]
    #                         # Moving WEST
    #                         if x != (Constants.M - 1):
    #                             # Probability of staying in y- and z-direction and only moving west!
    #                             P[pos, pos + t_step + x_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[0] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_WEST]
    #                         else: # x == M-1 going west means you are at x == 0 again; Note: moving DOWN in z-direction,
    #                             # STAY in y-direction and move WEST (x-direction) at the boundary!
    #                             P[pos, pos + t_step + x_step - Constants.M, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[0] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_WEST]
    #                         # Moving EAST
    #                         if x != 0:
    #                             # Probability of moving DOWN in z-direction, staying in y-direction and only moving east!
    #                             P[pos, pos + t_step - x_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[0] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_EAST]
    #                         else: # Moving EAST at boundary x == 0, will result in x == M-1!
    #                             P[pos, pos + t_step - x_step + Constants.M, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[0] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_EAST]
    #                         # Moving NORTH
    #                         if y != (Constants.N - 1): # Not and the Boundary N-1!
    #                             # Probability of moving NORTH (y-dir), going DOWN (z-dir), STAY (x-dir)
    #                             P[pos, pos + t_step + y_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[0] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_NORTH]
    #                         # Note: At the Boudnary N-1 go North is out of bounce which has a probability of 0!
                               
    #                         # Moving SOUTH
    #                         # only possible if y != 0 !!!
    #                         if y != 0:
    #                             P[pos, pos + t_step - y_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[0] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_SOUTH]
                                
    #                         # Move DOWN:
    #                         # STAY in every other direction x and y!
    #                         # Note: Still only possible if z != 0!
    #                         P[pos, pos + t_step - z_step, Constants.V_DOWN] = Constants.ALPHA \
    #                             * Constants.P_V_TRANSITION[1] \
    #                             * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_STAY]
    #                         # Moving WEST
    #                         if x != (Constants.M - 1):
    #                             # Only possible if M - 1
    #                             P[pos, pos + t_step - z_step + x_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[1] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_WEST]
    #                         else: # Return to x == 0 if going west from M - 1
    #                             P[pos, pos + t_step - z_step + x_step - Constants.M, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[1] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_WEST]
    #                         # Moving EAST
    #                         if x != 0:
    #                             P[pos, pos + t_step - z_step - x_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[1] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_EAST]
    #                         else: # Return again to M - 1!
    #                             P[pos, pos + t_step - z_step - x_step + Constants.M, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[1] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_EAST]
    #                         # Moving North
    #                         if y != (Constants.N - 1): # If not at upper boundary!
    #                             P[pos, pos + t_step - z_step + y_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[1] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_NORTH]
    #                         # Note: Else Moving North is not possible if at the boundary, henve P = 0!
                            
    #                         # Moving SOUTH
    #                         if y != 0:
    #                             P[pos, pos + t_step - z_step - y_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[1] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_SOUTH]
    #                         # Note: Else Moving South at the boundary is not possible, hence also here P = 0!

    #                     # if z == 0 Moving DOWN in z is not possible and would result in P = 0!

    #                 else: # t is at the boundary T - 1 hence it has to be set to t = 0 again!
    #                     # Now SAME code as before but just adapted for the case that t = T - 1
    #                     if z != 0: # Also I am not at the boundary z == 0!
    #                         # STAY in Z:
    #                         # STAY (Don't move at all!)
    #                         P[pos, pos + t_step - t_transform, Constants.V_DOWN] = Constants.ALPHA \
    #                             * Constants.P_V_TRANSITION[0] \
    #                             * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_STAY]
    #                         # Moving WEST
    #                         if x != (Constants.M - 1):
    #                             # Probability of staying in y- and z-direction and only moving west!
    #                             P[pos, pos + t_step - t_transform + x_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[0] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_WEST]
    #                         else: # x == M-1 going west means you are at x == 0 again; Note: moving DOWN in z-direction,
    #                             # STAY in y-direction and move WEST (x-direction) at the boundary!
    #                             P[pos, pos + t_step - t_transform + x_step - Constants.M, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[0] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_WEST]
    #                         # Moving EAST
    #                         if x != 0:
    #                             # Probability of moving DOWN in z-direction, staying in y-direction and only moving east!
    #                             P[pos, pos + t_step - t_transform - x_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[0] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_EAST]
    #                         else: # Moving EAST at boundary x == 0, will result in x == M-1!
    #                             P[pos, pos + t_step - t_transform - x_step + Constants.M, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[0] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_EAST]
    #                         # Moving NORTH
    #                         if y != (Constants.N - 1): # Not and the Boundary N-1!
    #                             # Probability of moving NORTH (y-dir), going DOWN (z-dir), STAY (x-dir)
    #                             P[pos, pos + t_step - t_transform + y_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[0] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_NORTH]
    #                         # Note: At the Boudnary N-1 go North is out of bounce which has a probability of 0!
                               
    #                         # Moving SOUTH
    #                         # only possible if y != 0 !!!
    #                         if y != 0:
    #                             P[pos, pos + t_step - t_transform - y_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[0] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_SOUTH]
                                
    #                         # Move DOWN:
    #                         # STAY in every other direction x and y!
    #                         # Note: Still only possible if z != 0!
    #                         P[pos, pos + t_step - t_transform - z_step, Constants.V_DOWN] = Constants.ALPHA \
    #                             * Constants.P_V_TRANSITION[1] \
    #                             * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_STAY]
    #                         # Moving WEST
    #                         if x != (Constants.M - 1):
    #                             # Only possible if M - 1
    #                             P[pos, pos + t_step - t_transform - z_step + x_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[1] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_WEST]
    #                         else: # Return to x == 0 if going west from M - 1
    #                             P[pos, pos + t_step - t_transform - z_step + x_step - Constants.M, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[1] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_WEST]
    #                         # Moving EAST
    #                         if x != 0:
    #                             P[pos, pos + t_step - t_transform - z_step - x_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[1] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_EAST]
    #                         else: # Return again to M - 1!
    #                             P[pos, pos + t_step - t_transform - z_step - x_step + Constants.M, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[1] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_EAST]
    #                         # Moving North
    #                         if y != (Constants.N - 1): # If not at upper boundary!
    #                             P[pos, pos + t_step - t_transform - z_step + y_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[1] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_NORTH]
    #                         # Note: Else Moving North is not possible if at the boundary, henve P = 0!
                            
    #                         # Moving SOUTH
    #                         if y != 0:
    #                             P[pos, pos + t_step - t_transform - z_step - y_step, Constants.V_DOWN] = Constants.ALPHA \
    #                                 * Constants.P_V_TRANSITION[1] \
    #                                 * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_SOUTH]
    #                         # Note: Else Moving South at the boundary is not possible, hence also here P = 0!


#########################################################################################
    # For all Cases handled!
    for u in range(input_space.size):
        for t in range(Constants.T): # Time
            for z in range(Constants.D): # z-Direction (Atmospheric level)
                for y in range(Constants.N): # y-Direction (Note: Boundary at y = 0 and y = N-1)
                    for x in range(Constants.M): # x-Direction (Note: Periodic!)
                        # Position:
                        pos = t * t_step + x * x_step + y * y_step + z * z_step

                        # Calculate the Probability of actually moving from i to j!
                        if z != u_boundary(u, Constants):
                            # STAY in Z:
                            # STAY (Don't move at all!)
                            P[pos, pos + t_step - t_transform(t, Constants, t_shift), u] = Constants.ALPHA \
                                * Constants.P_V_TRANSITION[0] \
                                * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_STAY]
                            # Moving WEST
                            if x != (Constants.M - 1):
                                # Probability of staying in y- and z-direction and only moving west!
                                P[pos, pos + t_step - t_transform(t, Constants, t_shift) + x_step, u] = Constants.ALPHA \
                                    * Constants.P_V_TRANSITION[0] \
                                    * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_WEST]
                            else: # x == M-1 going west means you are at x == 0 again; Note: moving DOWN in z-direction,
                                # STAY in y-direction and move WEST (x-direction) at the boundary!
                                P[pos, pos + t_step - t_transform(t, Constants, t_shift) + x_step - Constants.M, u] = Constants.ALPHA \
                                    * Constants.P_V_TRANSITION[0] \
                                    * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_WEST]
                            # Moving EAST
                            if x != 0:
                                # Probability of moving DOWN in z-direction, staying in y-direction and only moving east!
                                P[pos, pos + t_step - t_transform(t, Constants, t_shift) - x_step, u] = Constants.ALPHA \
                                    * Constants.P_V_TRANSITION[0] \
                                    * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_EAST]
                            else: # Moving EAST at boundary x == 0, will result in x == M-1!
                                P[pos, pos + t_step - t_transform(t, Constants, t_shift) - x_step + Constants.M, u] = Constants.ALPHA \
                                    * Constants.P_V_TRANSITION[0] \
                                    * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_EAST]
                            # Moving NORTH
                            if y != (Constants.N - 1): # Not and the Boundary N-1!
                                # Probability of moving NORTH (y-dir), going DOWN (z-dir), STAY (x-dir)
                                P[pos, pos + t_step - t_transform(t, Constants, t_shift) + y_step, u] = Constants.ALPHA \
                                    * Constants.P_V_TRANSITION[0] \
                                    * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_NORTH]
                            # Note: At the Boudnary N-1 go North is out of bounce which has a probability of 0!
                                
                            # Moving SOUTH
                            # only possible if y != 0 !!!
                            if y != 0:
                                P[pos, pos + t_step - t_transform(t, Constants, t_shift) - y_step, u] = Constants.ALPHA \
                                    * Constants.P_V_TRANSITION[0] \
                                    * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_SOUTH]

                            if u != Constants.V_STAY: 
                                # Move DOWN or UP if u != STAY
                                # STAY in every other direction x and y!
                                # Note: Still only possible if z != 0!
                                P[pos, pos + t_step - t_transform(t, Constants, t_shift) + move(u, Constants, z_step), u] = Constants.ALPHA \
                                    * Constants.P_V_TRANSITION[1] \
                                    * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_STAY]
                                # Moving WEST
                                if x != (Constants.M - 1):
                                    # Only possible if M - 1
                                    P[pos, pos + t_step - t_transform(t, Constants, t_shift) + move(u, Constants, z_step) + x_step, u] = Constants.ALPHA \
                                        * Constants.P_V_TRANSITION[1] \
                                        * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_WEST]
                                else: # Return to x == 0 if going west from M - 1
                                    P[pos, pos + t_step - t_transform(t, Constants, t_shift) + move(u, Constants, z_step) + x_step - Constants.M, u] = Constants.ALPHA \
                                        * Constants.P_V_TRANSITION[1] \
                                        * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_WEST]
                                # Moving EAST
                                if x != 0:
                                    P[pos, pos + t_step - t_transform(t, Constants, t_shift) + move(u, Constants, z_step) - x_step, u] = Constants.ALPHA \
                                        * Constants.P_V_TRANSITION[1] \
                                        * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_EAST]
                                else: # Return again to M - 1!
                                    P[pos, pos + t_step - t_transform(t, Constants, t_shift) + move(u, Constants, z_step) - x_step + Constants.M, u] = Constants.ALPHA \
                                        * Constants.P_V_TRANSITION[1] \
                                        * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_EAST]
                                # Moving North
                                if y != (Constants.N - 1): # If not at upper boundary!
                                    P[pos, pos + t_step - t_transform(t, Constants, t_shift) + move(u, Constants, z_step) + y_step, u] = Constants.ALPHA \
                                        * Constants.P_V_TRANSITION[1] \
                                        * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_NORTH]
                                # Note: Else Moving North is not possible if at the boundary, henve P = 0!
                                
                                # Moving SOUTH
                                if y != 0:
                                    P[pos, pos + t_step - t_transform(t, Constants, t_shift) + move(u, Constants, z_step) - y_step, u] = Constants.ALPHA \
                                        * Constants.P_V_TRANSITION[1] \
                                        * Constants.P_H_TRANSITION[z].P_WIND[Constants.H_SOUTH]
                                # Note: Else Moving South at the boundary is not possible, hence also here P = 0!

    return P

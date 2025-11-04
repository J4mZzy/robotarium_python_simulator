from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse

# Unused for now, will include later for speed.
# import quadprog as solver2

import itertools
import numpy as np
from scipy.special import comb

from rps.utilities.transformations import *

# Disable output of CVXOPT
options['show_progress'] = False
# Change default options of CVXOPT for faster solving
options['reltol'] = 1e-2 # was e-2
options['feastol'] = 1e-2 # was e-4
options['maxiters'] = 50 # default is 100

def create_single_integrator_barrier_certificate(barrier_gain=100, safety_radius=0.17, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system.  This function
    returns another function for optimization reasons.

    barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
    safety_radius: double (how far apart the agents will stay)
    magnitude_limit: how fast the robot can move linearly.

    -> function (the barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_radius, (int, float)), "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_radius >= 0.12, "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m) plus the distance to the look ahead point used in the diffeomorphism if that is being used. Recieved %r." % safety_radius
    assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    def f(dxi, x):
        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "In the function created by the create_single_integrator_barrier_certificate function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        
        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(comb(N, 2))*2
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2*np.identity(2*N)))
        h_min = np.inf

        count = 0

        #Centralized QP
        for i in range(N-1):
            for j in range(i+1,N):
                error = x[:, i] - x[:, j]
                # h = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)
                h = (error[0]/safety_radius)**2 + (error[1]/safety_radius)**2 - 1 

                A[count, (2*i, (2*i+1))] = -2*error/safety_radius**2
                A[count, (2*j, (2*j+1))] = 2*error/safety_radius**2
                b[count] = barrier_gain*np.power(h, 3)

                count += 1

                if h < h_min:
                    h_min = h

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

        f = -2*np.reshape(dxi, 2*N, order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F'), h_min

    return f

def create_single_integrator_barrier_certificate_decentralized(agent_index, barrier_gain=100, safety_radius=0.17, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system.  This function
    returns another function for optimization reasons.

    barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
    safety_radius: double (how far apart the agents will stay)
    magnitude_limit: how fast the robot can move linearly.

    -> function (the barrier certificate function)
    """

    #Check user input types
    assert isinstance(agent_index, int), "Agent index must be an integer. Received type %r." % type(agent_index).__name__
    assert isinstance(barrier_gain, (int, float)), "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_radius, (int, float)), "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_radius >= 0.12, "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m) plus the distance to the look ahead point used in the diffeomorphism if that is being used. Recieved %r." % safety_radius
    assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    def f(dxi, x):
        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
        # assert x.shape[1] == dxi.shape[1], "In the function created by the create_single_integrator_barrier_certificate function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        
        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(N-1)
        A = np.zeros((num_constraints, 2))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2*np.identity(2)))

        count = 0
        #Centralized QP
        for j in range(N):
            if j != agent_index:  # Skip the agent itself
                error = x[:, agent_index] - x[:, j]  # Relative position
                h = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)
                A[count, (0, 1)] = -2*error
                b[count] = barrier_gain*np.power(h, 3)

                count += 1

        # Threshold control inputs before QP
        norm = np.linalg.norm(dxi[:,agent_index], 2)
        # idxs_to_normalize = (norms > magnitude_limit)
        if norm > magnitude_limit:
            dxi *= magnitude_limit/norm

        f = -2*np.reshape(dxi[:,agent_index], 2, order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F')

    return f

def create_single_integrator_barrier_certificate_ellipse_decentralized(agent_index, barrier_gain=100, safety_a=0.17, safety_b=0.12, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system with an elliptical safety region.
    
    barrier_gain: double (controls how quickly agents can approach each other. lower = slower)
    safety_a: double (semi-major axis of the safety ellipse)
    safety_b: double (semi-minor axis of the safety ellipse)
    magnitude_limit: double (how fast the robot can move linearly)

    -> function (the barrier certificate function)
    """

    # Check user input types
    assert isinstance(agent_index, int), "Agent index must be an integer. Received type %r." % type(agent_index).__name__
    assert isinstance(barrier_gain, (int, float)), "The barrier gain must be an integer or float. Received type %r." % type(barrier_gain).__name__
    assert isinstance(safety_a, (int, float)), "The semi-major axis of the safety ellipse must be an integer or float. Received type %r." % type(safety_a).__name__
    assert isinstance(safety_b, (int, float)), "The semi-minor axis of the safety ellipse must be an integer or float. Received type %r." % type(safety_b).__name__
    assert isinstance(magnitude_limit, (int, float)), "The maximum linear velocity of the robot must be an integer or float. Received type %r." % type(magnitude_limit).__name__

    # Check user input ranges/sizes
    assert barrier_gain > 0, "The barrier gain must be positive. Received %r." % barrier_gain
    assert safety_a > 0 and safety_b > 0, "The semi-major and semi-minor axes must be positive. Received a: %r, b: %r." % (safety_a, safety_b)
    assert magnitude_limit > 0, "The maximum linear velocity of the robot must be positive. Received %r." % magnitude_limit

    def f(dxi, x,theta):
        # Check user input types
        assert isinstance(dxi, np.ndarray), "The single-integrator robot velocity command must be a numpy array. Received type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "The robot states must be a numpy array. Received type %r." % type(x).__name__

        # Check user input ranges/sizes
        assert x.shape[0] == 2, "The dimension of the single integrator robot states must be 2 ([x;y]). Received dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "The dimension of the robot single integrator velocity command must be 2 ([x_dot;y_dot]). Received dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "The number of robot states must be equal to the number of robot single integrator velocity commands. Received x: %r x %r, dxi: %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        # Initialize variables for computational savings
        N = dxi.shape[1]
        num_constraints = int((N - 1))
        A = np.zeros((num_constraints, 2))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2 * np.identity(2)))

        count = 0

        # decentralized QP
        for i in range(N):
            if i != agent_index:  # Skip the agent itself
                error = x[:, agent_index] - x[:, j]
                error_1 = (error[0]*np.cos(theta[agent_index])+error[1]*np.sin(theta[agent_index])) / safety_a
                error_2 = (error[0]*np.sin(theta[agent_index])-error[1]*np.cos(theta[agent_index])) / safety_b
                h = error_1**2 + error_2**2 - 1    

                # if np.mod(agent_index,2) == 0:
                    # theta[agent_index] = theta[agent_index] + np.pi/4
                A[count, 0] = -2 * ((error[0])*np.cos(theta[agent_index])+(error[1])*np.sin(theta[agent_index]))*np.cos(theta[agent_index])/ safety_a**2 - 2 * ((error[0])*np.sin(theta[agent_index])-(error[1])*np.cos(theta[agent_index]))*np.sin(theta[agent_index])/ safety_b**2
                A[count, 1] = -2 * ((error[0])*np.cos(theta[agent_index])+(error[1])*np.sin(theta[agent_index]))*np.sin(theta[agent_index])/ safety_a**2 - 2 * ((error[0])*np.sin(theta[agent_index])-(error[1])*np.cos(theta[agent_index]))*-np.cos(theta[agent_index])/ safety_b**2
                A[count, 2*j] = 2 * ((error[0])*np.cos(theta[i])+(error[1])*np.sin(theta[i]))*np.cos(theta[i])/ safety_a**2 + 2 * ((error[0])*np.sin(theta[i])-(error[1])*np.cos(theta[i]))*np.sin(theta[i])/ safety_b**2
                A[count, 2*j+1] = 2 * ((error[0])*np.cos(theta[i])+(error[1])*np.sin(theta[i]))*np.sin(theta[i])/ safety_a**2 + 2 * ((error[0])*np.sin(theta[i])-(error[1])*np.cos(theta[i]))*-np.cos(theta[i])/ safety_b**2
                # else:
                #     # theta[agent_index] = theta[agent_index] - np.pi/4
                #     A[count, 0] = -2 * ((error[0])*np.cos(theta[i])-(error[1])*np.sin(theta[i]))*np.cos(theta[i])/ safety_b**2 - 2 * ((error[0])*np.sin(theta[i])+(error[1])*np.cos(theta[i]))*np.sin(theta[i])/ safety_a**2
                #     A[count, 1] = -2 * ((error[0])*np.cos(theta[i])-(error[1])*np.sin(theta[i]))*-np.sin(theta[i])/ safety_b**2 - 2 * ((error[0])*np.sin(theta[i])+(error[1])*np.cos(theta[i]))*np.cos(theta[i])/ safety_a**2
                #     A[count, 2*j] = 2 * ((error[0])*np.cos(theta[i])-(error[1])*np.sin(theta[i]))*np.cos(theta[i])/ safety_b**2 + 2 * ((error[0])*np.sin(theta[i])+(error[1])*np.cos(theta[i]))*np.sin(theta[i])/ safety_a**2
                #     A[count, 2*j+1] = 2 * ((error[0])*np.cos(theta[i])-(error[1])*np.sin(theta[i]))*-np.sin(theta[i])/ safety_b**2 + 2 * ((error[0])*np.sin(theta[i])+(error[1])*np.cos(theta[i]))*np.cos(theta[i])/ safety_a**2

                b[count] = barrier_gain * h**3
                count += 1

        # Threshold control inputs before QP
        norm = np.linalg.norm(dxi[:,agent_index], 2)
        # idxs_to_normalize = (norms > magnitude_limit)
        if norm > magnitude_limit:
            dxi *= magnitude_limit/norm

        f = -2*np.reshape(dxi[:,agent_index], 2, order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F')

    return f




def create_single_integrator_barrier_certificate_ellipse(barrier_gain=100, safety_a=0.17, safety_b=0.12, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system with an elliptical safety region.
    
    barrier_gain: double (controls how quickly agents can approach each other. lower = slower)
    safety_a: double (semi-major axis of the safety ellipse)
    safety_b: double (semi-minor axis of the safety ellipse)
    magnitude_limit: double (how fast the robot can move linearly)

    -> function (the barrier certificate function)
    """

    # Check user input types
    assert isinstance(barrier_gain, (int, float)), "The barrier gain must be an integer or float. Received type %r." % type(barrier_gain).__name__
    assert isinstance(safety_a, (int, float)), "The semi-major axis of the safety ellipse must be an integer or float. Received type %r." % type(safety_a).__name__
    assert isinstance(safety_b, (int, float)), "The semi-minor axis of the safety ellipse must be an integer or float. Received type %r." % type(safety_b).__name__
    assert isinstance(magnitude_limit, (int, float)), "The maximum linear velocity of the robot must be an integer or float. Received type %r." % type(magnitude_limit).__name__

    # Check user input ranges/sizes
    assert barrier_gain > 0, "The barrier gain must be positive. Received %r." % barrier_gain
    assert safety_a > 0 and safety_b > 0, "The semi-major and semi-minor axes must be positive. Received a: %r, b: %r." % (safety_a, safety_b)
    assert magnitude_limit > 0, "The maximum linear velocity of the robot must be positive. Received %r." % magnitude_limit

    def f(dxi, x, theta):
        # Check user input types
        assert isinstance(dxi, np.ndarray), "The single-integrator robot velocity command must be a numpy array. Received type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "The robot states must be a numpy array. Received type %r." % type(x).__name__

        # Check user input ranges/sizes
        assert x.shape[0] == 2, "The dimension of the single integrator robot states must be 2 ([x;y]). Received dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "The dimension of the robot single integrator velocity command must be 2 ([x_dot;y_dot]). Received dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "The number of robot states must be equal to the number of robot single integrator velocity commands. Received x: %r x %r, dxi: %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        # Initialize variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(N * (N - 1))
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2 * np.identity(2*N)))
        h_min = np.inf

        count = 0
        # theta=theta+np.pi/4

        # Centralized QP
        for i in range(N-1):
            for j in range(i+1,N):
                error = x[:, i] - x[:, j]

                ex, ey = error[0], error[1]
                c = np.cos(theta[i]); s = np.sin(theta[i])
                ## Rotation 
                u =  c*ex - s*ey
                v =  s*ex + c*ey
                h = (u/safety_a)**2 + (v/safety_b)**2 - 1.0

                # h_ellip dot
                # h_ellip_dot1 = 2 * ((error[0])*np.cos(theta[i])+(error[1])*np.sin(theta[i]))*np.cos(theta[i])/ safety_a**2 + 2 * ((error[0])*np.sin(theta[i])-(error[1])*np.cos(theta[i]))*np.sin(theta[i])/ safety_b**2
                # h_ellip_dot2 = 2 * ((error[0])*np.cos(theta[i])+(error[1])*np.sin(theta[i]))*np.sin(theta[i])/ safety_a**2 + 2 * ((error[0])*np.sin(theta[i])-(error[1])*np.cos(theta[i]))*-np.cos(theta[i])/ safety_b**2

                inv_a2 = 1.0 / (safety_a**2)
                inv_b2 = 1.0 / (safety_b**2)
                h_ellip_dot1 = 2.0 * (u * c * inv_a2 + v * s * inv_b2)        # ∂/∂ex
                h_ellip_dot2 = 2.0 * (-u * s * inv_a2 + v * c * inv_b2)       # ∂/∂ey

                A[count, 2*i] = -h_ellip_dot1
                A[count, 2*i+1] = -h_ellip_dot2
                A[count, 2*j] = h_ellip_dot1
                A[count, 2*j+1] = h_ellip_dot2
        

                b[count] = barrier_gain * h**3
                count += 1

                if h < h_min:
                    h_min = h


        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, axis=0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit / norms[idxs_to_normalize]
    
        f = -2 * np.reshape(dxi, (2*N,), order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F'), h_min

    return f

def create_single_integrator_barrier_certificate_square(barrier_gain=100, safety_width=0.34, norm = 3, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system.  This function
    returns another function for optimization reasons.

    barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
    safety_radius: double (how far apart the agents will stay)
    magnitude_limit: how fast the robot can move linearly.

    -> function (the barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_width, (int, float)), "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_width).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_width >= 0.24, "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m) plus the distance to the look ahead point used in the diffeomorphism if that is being used. Recieved %r." % safety_width
    assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    def f(dxi, x, theta):
        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "In the function created by the create_single_integrator_barrier_certificate function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        
        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(comb(N, 2))*2
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2*np.identity(2*N)))
        p = norm

        count = 0
        #Centralized QP
        for i in range(N-1):
            for j in range(i+1,N):
                error = x[:, i] - x[:, j]
                ex, ey = error[0], error[1]
                c = np.cos(theta[i]); s = np.sin(theta[i])

                # --- rotate by +θ: [u;v] = [[c,-s],[s,c]] [ex;ey]
                u =  c*ex - s*ey
                v =  s*ex + c*ey
                
                ## p-norm h = ||R(theta)*x||_p - r
                # h = (np.abs(error_1)**p + np.abs(error_2)**p)**(1.0/p) - safety_width/2.0
                h = (np.abs(u)**3 + np.abs(v)**3)**(1.0/3.0) - safety_width/2.0
                # h_square_dot1 = (np.sin(theta[i])*error_2*np.abs(error_2) + np.cos(theta[i])*error_1*np.abs(error_1))/(error_2**2*np.abs(error_2) + error_1**2*np.abs(error_1))**(2/3)
                # h_square_dot2 = (np.sin(theta[i])*error_1*np.abs(error_1) - np.cos(theta[i])*error_2*np.abs(error_2))/(error_1**2*np.abs(error_1) + error_2**2*np.abs(error_2))**(2/3)
                
                # 3-norm “square”
                den = (np.abs(u)**3 + np.abs(v)**3)**(2.0/3.0)
                uu = u * np.abs(u)
                vv = v * np.abs(v)
                h_square_dot1 = (c*uu + s*vv) / den                            # ∂/∂ex
                h_square_dot2 = (-s*uu + c*vv) / den                           # ∂/∂ey


                A[count, 2*i] = -h_square_dot1
                A[count, 2*i+1] = -h_square_dot2
                A[count, 2*j] = h_square_dot1
                A[count, 2*j+1] = h_square_dot2

                b[count] = barrier_gain*np.power(h, 3)
                count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

        f = -2*np.reshape(dxi, 2*N, order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F')

    return f

def create_single_integrator_barrier_certificate_triangle(barrier_gain=100, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system.  This function
    returns another function for optimization reasons.

    barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
    safety_radius: double (how far apart the agents will stay)
    magnitude_limit: how fast the robot can move linearly.

    -> function (the barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    def f(dxi, x, theta):
        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "In the function created by the create_single_integrator_barrier_certificate function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        
        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(comb(N, 2))*2
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2*np.identity(2*N)))

        count = 0
        #Centralized QP
        for i in range(N-1):
            for j in range(i+1,N):
                error = x[:, i] - x[:, j]

                ex, ey = error[0], error[1]
                c = np.cos(theta[i]); s = np.sin(theta[i])

                # --- rotate by +θ: [u;v] = [[c,-s],[s,c]] [ex;ey]
                u =  c*ex - s*ey
                v =  s*ex + c*ey

                # triangle (log-sum-exp), numerically stable
                rt3 = np.sqrt(3.0)
                L1 = 4*u + 4*rt3*v
                L2 = -8*u
                L3 = 4*u - 4*rt3*v
                M  = np.maximum.reduce([L1, L2, L3])
                w1 = np.exp(L1 - M); w2 = np.exp(L2 - M); w3 = np.exp(L3 - M)
                S  = w1 + w2 + w3
                h = (3.0/5.0) * (M + np.log(S)) - 1.0

                # triangle: first ∂h/∂u, ∂h/∂v via softmax weights, then chain to (ex,ey)
                a1 = w1 / S; a2 = w2 / S; a3 = w3 / S
                h_u = (3.0/5.0) * (4*a1 - 8*a2 + 4*a3)
                h_v = (3.0/5.0) * (4*rt3*(a1 - a3))
                h_tri_dot1 = c*h_u + s*h_v                                      # ∂/∂ex
                h_tri_dot2 = -s*h_u + c*h_v                                     # ∂/∂ey
                ## Rotation
                # error_1 = (error[0]*np.cos(theta[i])+error[1]*np.sin(theta[i])) 
                # error_2 = (error[0]*np.sin(theta[i])-error[1]*np.cos(theta[i])) 

                # h = 3/5 * np.log(np.exp(4*error_1+4*np.sqrt(3)*error_2) + np.exp(-8*error_1) + np.exp(4*error_1-4*np.sqrt(3)*error_2))- 1

                # h_tri_dot1 = 12*((np.sqrt(3)*np.sin(theta[i])+np.cos(theta[i]))*np.exp(8*np.sqrt(3)*error_2 + 12*error_1) 
                #               - 2*np.cos(theta[i])*np.exp(4*np.sqrt(3)*error_2)-(np.sqrt(3)*np.sin(theta[i])-np.cos(theta[i]))*np.exp(12*error_1))/ (5*(np.exp(8*np.sqrt(3)*error_2+12*error_1)
                #               + np.exp(4*np.sqrt(3)*error_2) + np.exp(12*error_1)))
                # h_tri_dot2 = 12*((np.sin(theta[i])-np.sqrt(3)*np.cos(theta[i]))*np.exp(8*np.sqrt(3)*error_2 + 12*error_1)
                #               - 2*np.sin(theta[i])*np.exp(4*np.sqrt(3)*error_2)+(np.sin(theta[i])+np.sqrt(3)*np.cos(theta[i]))*np.exp(12*error_1))/ (5*((np.exp(8*np.sqrt(3)*error_2)+1)*np.exp(12*error_1)
                #               + np.exp(4*np.sqrt(3)*error_2)))
                
                
                # x_1 = error[0] 
                # x_2 = error[1] 

                # h = 0.6 * np.log(np.exp(4*x_1+4*np.sqrt(3)*x_2) + np.exp(-8*x_1) + np.exp(4*x_1-4*np.sqrt(3)*x_2))- 1

                # h_tri_dot1 = 1.2 * ((2*np.exp(8*np.sqrt(3)*x_2) + 2)*np.exp(12*x_1) - 4*np.exp(4*np.sqrt(3)*x_2))/((np.exp(8*np.sqrt(3)*x_2) + 1)*np.exp(12*x_1) + np.exp(4*np.sqrt(3)*x_2))
                # h_tri_dot2 =  1.2 * 2*np.sqrt(3)*np.exp(12*x_1)*(np.exp(8*np.sqrt(3)*x_2) - 1)/ (np.exp(8*np.sqrt(3)*x_2+12*x_1) + np.exp(4*np.sqrt(3)*x_2) + np.exp(12*x_1))


                A[count, 2*i] = -h_tri_dot1
                A[count, 2*i+1] = -h_tri_dot2
                A[count, 2*j] = h_tri_dot1
                A[count, 2*j+1] = h_tri_dot2

                b[count] = barrier_gain*np.power(h, 3)
                count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

        f = -2*np.reshape(dxi, 2*N, order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F')

    return f





def create_single_integrator_barrier_certificate_time_varying(Delta, lamb,target_shape, Delta_dot, safety_radius=0.17, barrier_gain=100, safety_a=0.17, safety_b=0.12, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system with circle to elliptical time varying safety region.
    
    barrier_gain: double (controls how quickly agents can approach each other. lower = slower)
    safety_a: double (semi-major axis of the safety ellipse)
    safety_b: double (semi-minor axis of the safety ellipse)
    magnitude_limit: double (how fast the robot can move linearly)

    -> function (the barrier certificate function)
    """

    # Check user input types
    assert isinstance(barrier_gain, (int, float)), "The barrier gain must be an integer or float. Received type %r." % type(barrier_gain).__name__
    assert isinstance(safety_a, (int, float)), "The semi-major axis of the safety ellipse must be an integer or float. Received type %r." % type(safety_a).__name__
    assert isinstance(safety_b, (int, float)), "The semi-minor axis of the safety ellipse must be an integer or float. Received type %r." % type(safety_b).__name__
    assert isinstance(magnitude_limit, (int, float)), "The maximum linear velocity of the robot must be an integer or float. Received type %r." % type(magnitude_limit).__name__

    # Check user input ranges/sizes
    assert barrier_gain > 0, "The barrier gain must be positive. Received %r." % barrier_gain
    assert safety_a > 0 and safety_b > 0, "The semi-major and semi-minor axes must be positive. Received a: %r, b: %r." % (safety_a, safety_b)
    assert magnitude_limit > 0, "The maximum linear velocity of the robot must be positive. Received %r." % magnitude_limit

    def f(dxi, x,theta):
        # Check user input types
        assert isinstance(dxi, np.ndarray), "The single-integrator robot velocity command must be a numpy array. Received type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "The robot states must be a numpy array. Received type %r." % type(x).__name__

        # Check user input ranges/sizes
        assert x.shape[0] == 2, "The dimension of the single integrator robot states must be 2 ([x;y]). Received dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "The dimension of the robot single integrator velocity command must be 2 ([x_dot;y_dot]). Received dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "The number of robot states must be equal to the number of robot single integrator velocity commands. Received x: %r x %r, dxi: %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        # Initialize variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(N * (N - 1)/2)
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2 * np.identity(2*N)))
        
        count = 0

        # Centralized QP
        # for i in range(N):
        #     for j in range(N):
        #         if i == j:
        #             continue
        #         error = x[:, i] - x[:, j]
        for i in range(N-1):
            for j in range(i+1,N):
                error = x[:, i] - x[:, j]
                ex, ey = error[0], error[1]
                c = np.cos(theta[i]); s = np.sin(theta[i])

                # --- rotate by +θ: [u;v] = [[c,-s],[s,c]] [ex;ey]
                u =  c*ex - s*ey
                v =  s*ex + c*ey
                ###################################### library ########################################
                # circle
                h_circ  = (ex/safety_radius)**2 + (ey/safety_radius)**2 - 1.0

                # ellipse
                h_ellip = (u/safety_a)**2 + (v/safety_b)**2 - 1.0

                # 3-norm “square”
                safety_width = 0.4  # if not set elsewhere
                h_square = (np.abs(u)**3 + np.abs(v)**3)**(1.0/3.0) - safety_width/2.0

                # triangle (log-sum-exp), numerically stable
                rt3 = np.sqrt(3.0)
                L1 = 4*u + 4*rt3*v
                L2 = -8*u
                L3 = 4*u - 4*rt3*v
                M  = np.maximum.reduce([L1, L2, L3])
                w1 = np.exp(L1 - M); w2 = np.exp(L2 - M); w3 = np.exp(L3 - M)
                S  = w1 + w2 + w3
                h_tri = (3.0/5.0) * (M + np.log(S)) - 1.0

                # convex combo
                h_cur = lamb[0]*h_circ + lamb[1]*h_ellip + lamb[2]*h_tri + lamb[3]*h_square

                # ---------------- gradients wrt (ex, ey) ----------------
                # circle
                h_circ_dot1 = 2.0 * ex / (safety_radius**2)
                h_circ_dot2 = 2.0 * ey / (safety_radius**2)

                # ellipse  (du/dex=c, du/dey=-s, dv/dex=s, dv/dey=c)
                inv_a2 = 1.0 / (safety_a**2)
                inv_b2 = 1.0 / (safety_b**2)
                h_ellip_dot1 = 2.0 * (u * c * inv_a2 + v * s * inv_b2)        # ∂/∂ex
                h_ellip_dot2 = 2.0 * (-u * s * inv_a2 + v * c * inv_b2)       # ∂/∂ey

                # 3-norm “square”
                eps = 1e-12
                den = (np.abs(u)**3 + np.abs(v)**3 + eps)**(2.0/3.0)
                uu = u * np.abs(u)
                vv = v * np.abs(v)
                h_square_dot1 = (c*uu + s*vv) / den                            # ∂/∂ex
                h_square_dot2 = (-s*uu + c*vv) / den                           # ∂/∂ey

                # triangle: first ∂h/∂u, ∂h/∂v via softmax weights, then chain to (ex,ey)
                a1 = w1 / S; a2 = w2 / S; a3 = w3 / S
                h_u = (3.0/5.0) * (4*a1 - 8*a2 + 4*a3)
                h_v = (3.0/5.0) * (4*rt3*(a1 - a3))
                h_tri_dot1 = c*h_u + s*h_v                                      # ∂/∂ex
                h_tri_dot2 = -s*h_u + c*h_v                                     # ∂/∂ey

                # blended gradient (for your current h_cur)
                h_cur_dot1 = lamb[0]*h_circ_dot1 + lamb[1]*h_ellip_dot1 + lamb[2]*h_tri_dot1 + lamb[3]*h_square_dot1
                h_cur_dot2 = lamb[0]*h_circ_dot2 + lamb[1]*h_ellip_dot2 + lamb[2]*h_tri_dot2 + lamb[3]*h_square_dot2

                # if 1 switching current to circle, if 2 switching current to ellipse
                if target_shape == 1:
                    h_tv = (1-Delta) * h_cur + Delta * h_circ 
                    # for time derivative 
                    diff = h_circ - h_cur

                    A[count, 2*i] = -((1-Delta) * h_cur_dot1 + Delta * h_circ_dot1)
                    A[count, 2*i+1] = -((1-Delta) * h_cur_dot2 + Delta * h_circ_dot2)
                    A[count, 2*j] = (1-Delta) * h_cur_dot1 + Delta * h_circ_dot1
                    A[count, 2*j+1] = (1-Delta) * h_cur_dot2 + Delta * h_circ_dot2
                elif target_shape == 2:
                    h_tv = (1-Delta) * h_cur + Delta * h_ellip 
                    # for time derivative 
                    diff = h_ellip - h_cur

                    A[count, 2*i] = -((1-Delta) * h_cur_dot1 + Delta * h_ellip_dot1)
                    A[count, 2*i+1] = -((1-Delta) * h_cur_dot2 + Delta * h_ellip_dot2)
                    A[count, 2*j] = (1-Delta) * h_cur_dot1 + Delta * h_ellip_dot1
                    A[count, 2*j+1] = (1-Delta) * h_cur_dot2 + Delta * h_ellip_dot2
                elif target_shape == 3:
                    h_tv = (1-Delta) * h_cur + Delta * h_tri 
                    # for time derivative 
                    diff = h_tri - h_cur

                    A[count, 2*i] = -((1-Delta) * h_cur_dot1 + Delta * h_tri_dot1)
                    A[count, 2*i+1] = -((1-Delta) * h_cur_dot2 + Delta * h_tri_dot2)
                    A[count, 2*j] = (1-Delta) * h_cur_dot1 + Delta * h_tri_dot1
                    A[count, 2*j+1] = (1-Delta) * h_cur_dot2 + Delta * h_tri_dot2
                elif target_shape == 4:
                    h_tv = (1-Delta) * h_cur + Delta * h_square 
                    # for time derivative 
                    diff = h_square - h_cur

                    A[count, 2*i] = -((1-Delta) * h_cur_dot1 + Delta * h_square_dot1)
                    A[count, 2*i+1] = -((1-Delta) * h_cur_dot2 + Delta * h_square_dot2)
                    A[count, 2*j] = (1-Delta) * h_cur_dot1 + Delta * h_square_dot1
                    A[count, 2*j+1] = (1-Delta) * h_cur_dot2 + Delta * h_square_dot2
                
                # class k function and time derivative in delta 
                b[count] = barrier_gain * h_tv**3 + Delta_dot * diff
                #  Delta_dot * diff
                count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, axis=0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit / norms[idxs_to_normalize]

        f = -2 * np.reshape(dxi, (2*N,), order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        result = np.reshape(result, (2, -1), order='F')
        
        norms2 = np.linalg.norm(result, 2, axis=0)
        idxs_to_normalize2 = (norms2 > magnitude_limit)
        result[:, idxs_to_normalize2] *= magnitude_limit / norms2[idxs_to_normalize2]

        return result

    return f

def create_single_integrator_barrier_certificate_diamond(barrier_gain=100, safety_radius=0.17, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system.  This function
    returns another function for optimization reasons.

    barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
    safety_radius: double (how far apart the agents will stay)
    magnitude_limit: how fast the robot can move linearly.

    -> function (the barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_radius, (int, float)), "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_radius >= 0.12, "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m) plus the distance to the look ahead point used in the diffeomorphism if that is being used. Recieved %r." % safety_radius
    assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    def f(dxi, x):
        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "In the function created by the create_single_integrator_barrier_certificate function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        
        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(comb(N, 2))*2
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2*np.identity(2*N)))

        count = 0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                error = x[:, i] - x[:, j]
                h = (np.abs(error[0]) + np.abs(error[1]))-safety_radius #l_1 norm

                A[count, (2*i, (2*i+1))] = -np.sign(error)
                A[count, (2*j, (2*j+1))] = np.sign(error)
                b[count] = barrier_gain*np.power(h, 3)

                count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

        f = -1*np.reshape(dxi, 2*N, order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F')

    return f

def create_single_integrator_barrier_certificate_with_boundary(barrier_gain=100, safety_radius=0.17, magnitude_limit=0.2, boundary_points = np.array([-1.6, 1.6, -1.2, 1.2])):
    """Creates a barrier certificate for a single-integrator system with a rectangular boundary included.  This function
    returns another function for optimization reasons.

    barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
    safety_radius: double (how far apart the agents will stay)
    magnitude_limit: how fast the robot can move linearly.

    -> function (the barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_radius, (int, float)), "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_radius >= 0.12, "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m) plus the distance to the look ahead point used in the diffeomorphism if that is being used. Recieved %r." % safety_radius
    assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    def f(dxi, x):
        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "In the function created by the create_single_integrator_barrier_certificate function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        
        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(comb(N, 2))*2 + 4*N
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        #H = sparse(matrix(2*np.identity(2*N)))
        H = 2*np.identity(2*N)

        count = 0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                error = x[:, i] - x[:, j]
                h = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)

                A[count, (2*i, (2*i+1))] = -2 * error
                A[count, (2*j, (2*j+1))] = 2 * error
                b[count] = barrier_gain*np.power(h, 3) #    cubic
                # b[count] = barrier_gain*h

                count += 1
        
        for k in range(N):
            boundary_safety_radius = safety_radius - 0.1
            #Pos Y
            A[count, (2*k, 2*k+1)] = np.array([0,1])
            b[count] = 0.4*barrier_gain*(boundary_points[3] - boundary_safety_radius/2 - x[1,k])**3  # cubic
            # b[count] = 0.4*barrier_gain*(boundary_points[3] - safety_radius/2 - x[1,k])
            count += 1

            #Neg Y
            A[count, (2*k, 2*k+1)] = -np.array([0,1])
            b[count] = 0.4*barrier_gain*(-boundary_points[2] - boundary_safety_radius/2 + x[1,k])**3
            # b[count] = 0.4*barrier_gain*(-boundary_points[2] - safety_radius/2 + x[1,k])
            count += 1

            #Pos X
            A[count, (2*k, 2*k+1)] = np.array([1,0])
            b[count] = 0.4*barrier_gain*(boundary_points[1] - boundary_safety_radius/2 - x[0,k])**3
            # b[count] = 0.4*barrier_gain*(boundary_points[1] - safety_radius/2 - x[0,k])
            count += 1

            #Neg X
            A[count, (2*k, 2*k+1)] = -np.array([1,0])
            b[count] = 0.4*barrier_gain*(-boundary_points[0] - boundary_safety_radius/2 + x[0,k])**3
            # b[count] = 0.4*barrier_gain*(-boundary_points[0] - safety_radius/2 + x[0,k])
            count += 1
        
        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

        f = -2*np.reshape(dxi, (2*N,1), order='F')
        b = np.reshape(b, (count,1), order='F')
        result = qp(matrix(H), matrix(f), matrix(A), matrix(b))['x']
        #result = solver2.solve_qp(H, f, A, b, 0)[0]

        return np.reshape(result, (2, N), order='F')

    return f


def create_single_integrator_barrier_certificate_with_obstacles(barrier_gain=100, safety_radius=0.17, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system with a rectangular boundary included.  This function
    returns another function for optimization reasons.

    barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
    safety_radius: double (how far apart the agents will stay)
    magnitude_limit: how fast the robot can move linearly.

    -> function (the barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_radius, (int, float)), "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_radius >= 0.12, "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m) plus the distance to the look ahead point used in the diffeomorphism if that is being used. Recieved %r." % safety_radius
    assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    def f(dxi, x):
        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "In the function created by the create_single_integrator_barrier_certificate function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        obstacle_centers = np.array([[-0.6,  0.6],
                                 [0,  0]], dtype=float)  # shape (2, 2)
        obstacle_radius = 0.25 # fixed radius
        K = 2 # number of obstacles

        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(comb(N, 2)) + N * K
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        #H = sparse(matrix(2*np.identity(2*N)))
        H = 2*np.identity(2*N)
        h_min = np.inf

        count = 0
        for i in range(N-1):
            for j in range(i+1,N):
                error = x[:, i] - x[:, j]
                # h = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)
                h = (error[0]/safety_radius)**2 + (error[1]/safety_radius)**2 - 1 

                A[count, (2*i, (2*i+1))] = -2 * error/safety_radius**2
                A[count, (2*j, (2*j+1))] = 2 * error/safety_radius**2
                b[count] = barrier_gain*np.power(h, 3) #    cubic
                # b[count] = barrier_gain*h
                count += 1
            if h < h_min:
                h_min = h
        
        for i in range(N):
                    xi = x[:, i]
                    for k in range(K):
                        ck = obstacle_centers[:, k]
                        err = xi - ck
                        h_obs = (err[0] * err[0] + err[1] * err[1]) - np.power(obstacle_radius, 2)

                        # Only agent i's control appears
                        A[count, (2 * i, 2 * i + 1)] = -2 * err
                        b[count] = 0.5 * np.power(h_obs, 3) # fixed gain
                        count += 1
        
        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

        f = -2*np.reshape(dxi, (2*N,1), order='F')
        b = np.reshape(b, (count,1), order='F')
        result = qp(matrix(H), matrix(f), matrix(A), matrix(b))['x']
        #result = solver2.solve_qp(H, f, A, b, 0)[0]

        return np.reshape(result, (2, N), order='F'), h_min

    return f


def create_single_integrator_barrier_certificate_ellipse_with_obstacles(barrier_gain=100, safety_a=0.17, safety_b=0.12, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system with an elliptical safety region.
    
    barrier_gain: double (controls how quickly agents can approach each other. lower = slower)
    safety_a: double (semi-major axis of the safety ellipse)
    safety_b: double (semi-minor axis of the safety ellipse)
    magnitude_limit: double (how fast the robot can move linearly)

    -> function (the barrier certificate function)
    """

    # Check user input types
    assert isinstance(barrier_gain, (int, float)), "The barrier gain must be an integer or float. Received type %r." % type(barrier_gain).__name__
    assert isinstance(safety_a, (int, float)), "The semi-major axis of the safety ellipse must be an integer or float. Received type %r." % type(safety_a).__name__
    assert isinstance(safety_b, (int, float)), "The semi-minor axis of the safety ellipse must be an integer or float. Received type %r." % type(safety_b).__name__
    assert isinstance(magnitude_limit, (int, float)), "The maximum linear velocity of the robot must be an integer or float. Received type %r." % type(magnitude_limit).__name__

    # Check user input ranges/sizes
    assert barrier_gain > 0, "The barrier gain must be positive. Received %r." % barrier_gain
    assert safety_a > 0 and safety_b > 0, "The semi-major and semi-minor axes must be positive. Received a: %r, b: %r." % (safety_a, safety_b)
    assert magnitude_limit > 0, "The maximum linear velocity of the robot must be positive. Received %r." % magnitude_limit

    def f(dxi, x, theta):
        # Check user input types
        assert isinstance(dxi, np.ndarray), "The single-integrator robot velocity command must be a numpy array. Received type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "The robot states must be a numpy array. Received type %r." % type(x).__name__

        # Check user input ranges/sizes
        assert x.shape[0] == 2, "The dimension of the single integrator robot states must be 2 ([x;y]). Received dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "The dimension of the robot single integrator velocity command must be 2 ([x_dot;y_dot]). Received dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "The number of robot states must be equal to the number of robot single integrator velocity commands. Received x: %r x %r, dxi: %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        obstacle_centers = np.array([[-0.6,  0.6],
                                 [0,  0]], dtype=float)  # shape (2, 2)
        obstacle_radius = 0.25 # fixed radius
        K = 2 # number of obstacles

        # Initialize variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(N * (N - 1)) + N*K
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2 * np.identity(2*N)))
        h_min = np.inf

        count = 0
        # theta=theta+np.pi/4

        # Centralized QP
        for i in range(N-1):
            for j in range(i+1,N):
                error = x[:, i] - x[:, j]

                ex, ey = error[0], error[1]
                c = np.cos(theta[i]); s = np.sin(theta[i])
                ## Rotation 
                u =  c*ex - s*ey
                v =  s*ex + c*ey
                h = (u/safety_a)**2 + (v/safety_b)**2 - 1.0

                # h_ellip dot
                # h_ellip_dot1 = 2 * ((error[0])*np.cos(theta[i])+(error[1])*np.sin(theta[i]))*np.cos(theta[i])/ safety_a**2 + 2 * ((error[0])*np.sin(theta[i])-(error[1])*np.cos(theta[i]))*np.sin(theta[i])/ safety_b**2
                # h_ellip_dot2 = 2 * ((error[0])*np.cos(theta[i])+(error[1])*np.sin(theta[i]))*np.sin(theta[i])/ safety_a**2 + 2 * ((error[0])*np.sin(theta[i])-(error[1])*np.cos(theta[i]))*-np.cos(theta[i])/ safety_b**2

                inv_a2 = 1.0 / (safety_a**2)
                inv_b2 = 1.0 / (safety_b**2)
                h_ellip_dot1 = 2.0 * (u * c * inv_a2 + v * s * inv_b2)        # ∂/∂ex
                h_ellip_dot2 = 2.0 * (-u * s * inv_a2 + v * c * inv_b2)       # ∂/∂ey

                # if np.mod(i,2) == 0:
                A[count, 2*i] = -h_ellip_dot1
                A[count, 2*i+1] = -h_ellip_dot2
                A[count, 2*j] = h_ellip_dot1
                A[count, 2*j+1] = h_ellip_dot2

                b[count] = barrier_gain * h**3
                count += 1

                if h < h_min:
                    h_min = h


        for i in range(N):
                    xi = x[:, i]
                    for k in range(K):
                        ck = obstacle_centers[:, k]
                        err = xi - ck
                        h_obs = (err[0] * err[0] + err[1] * err[1]) - np.power(obstacle_radius, 2)

                        # Only agent i's control appears
                        A[count, (2 * i, 2 * i + 1)] = -2 * err
                        b[count] = 0.5 * np.power(h_obs, 3) # fixed gain
                        count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, axis=0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit / norms[idxs_to_normalize]
    
        f = -2 * np.reshape(dxi, (2*N,), order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F'), h_min

    return f

def create_single_integrator_barrier_certificate_triangle_with_obstacles(barrier_gain=100, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system.  This function
    returns another function for optimization reasons.

    barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
    safety_radius: double (how far apart the agents will stay)
    magnitude_limit: how fast the robot can move linearly.

    -> function (the barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit

    def f(dxi, x, theta):
        # Check user input types
        assert isinstance(dxi, np.ndarray), "The single-integrator robot velocity command must be a numpy array. Received type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "The robot states must be a numpy array. Received type %r." % type(x).__name__

        # Check user input ranges/sizes
        assert x.shape[0] == 2, "The dimension of the single integrator robot states must be 2 ([x;y]). Received dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "The dimension of the robot single integrator velocity command must be 2 ([x_dot;y_dot]). Received dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "The number of robot states must be equal to the number of robot single integrator velocity commands. Received x: %r x %r, dxi: %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        obstacle_centers = np.array([[-0.6,  0.6],
                                 [0,  0]], dtype=float)  # shape (2, 2)
        obstacle_radius = 0.25 # fixed radius
        K = 2 # number of obstacles

        # Initialize variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(N * (N - 1)/2) + N*K
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2 * np.identity(2*N)))
        h_min = np.inf

        count = 0
        # theta=theta+np.pi/4

        # Centralized QP
        for i in range(N-1):
            for j in range(i+1,N):
                error = x[:, i] - x[:, j]
                ex, ey = error[0], error[1]
                c = np.cos(theta[i]); s = np.sin(theta[i])

                # --- rotate by +θ: [u;v] = [[c,-s],[s,c]] [ex;ey]
                u =  c*ex - s*ey
                v =  s*ex + c*ey

                # triangle (log-sum-exp), numerically stable
                rt3 = np.sqrt(3.0)
                L1 = 4*u + 4*rt3*v
                L2 = -8*u
                L3 = 4*u - 4*rt3*v
                M  = np.maximum.reduce([L1, L2, L3])
                w1 = np.exp(L1 - M); w2 = np.exp(L2 - M); w3 = np.exp(L3 - M)
                S  = w1 + w2 + w3
                h = (3.0/5.0) * (M + np.log(S)) - 1.0

                # triangle: first ∂h/∂u, ∂h/∂v via softmax weights, then chain to (ex,ey)
                a1 = w1 / S; a2 = w2 / S; a3 = w3 / S
                h_u = (3.0/5.0) * (4*a1 - 8*a2 + 4*a3)
                h_v = (3.0/5.0) * (4*rt3*(a1 - a3))
                h_tri_dot1 = c*h_u + s*h_v                                      # ∂/∂ex
                h_tri_dot2 = -s*h_u + c*h_v                                     # ∂/∂ey

                # if np.mod(i,2) == 0:
                A[count, 2*i] = -h_tri_dot1
                A[count, 2*i+1] = -h_tri_dot2
                A[count, 2*j] = h_tri_dot1
                A[count, 2*j+1] = h_tri_dot2

                b[count] = barrier_gain * h**3
                count += 1

                if h < h_min:
                    h_min = h


        for i in range(N):
                    xi = x[:, i]
                    for k in range(K):
                        ck = obstacle_centers[:, k]
                        err = xi - ck
                        h_obs = (err[0] * err[0] + err[1] * err[1]) - np.power(obstacle_radius, 2)

                        # Only agent i's control appears
                        A[count, (2 * i, 2 * i + 1)] = -2 * err
                        b[count] = 0.5 * np.power(h_obs, 3) # fixed gain
                        count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, axis=0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit / norms[idxs_to_normalize]
    
        f = -2 * np.reshape(dxi, (2*N,), order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F'), h_min

    return f


def create_single_integrator_barrier_certificate_square_with_obstacles(barrier_gain=100, safety_width=0.34, norm = 3, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system.  This function
    returns another function for optimization reasons.

    barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
    safety_radius: double (how far apart the agents will stay)
    magnitude_limit: how fast the robot can move linearly.

    -> function (the barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_width, (int, float)), "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_width).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_width >= 0.24, "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m) plus the distance to the look ahead point used in the diffeomorphism if that is being used. Recieved %r." % safety_width
    assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit

    obstacle_centers = np.array([[-0.6,  0.6],
                                 [0,  0]], dtype=float)  # shape (2, 2)
    obstacle_radius = 0.25 # fixed radius
    K = 2 # number of obstacles

    def f(dxi, x, theta):
        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "In the function created by the create_single_integrator_barrier_certificate function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        
        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(comb(N, 2))+N*K
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2*np.identity(2*N)))
        p = norm
        h_min = np.inf

        count = 0
        #Centralized QP
        for i in range(N-1):
            for j in range(i+1,N):
                error = x[:, i] - x[:, j]
                ex, ey = error[0], error[1]
                c = np.cos(theta[i]); s = np.sin(theta[i])

                # --- rotate by +θ: [u;v] = [[c,-s],[s,c]] [ex;ey]
                u =  c*ex - s*ey
                v =  s*ex + c*ey
                
                ## p-norm h = ||R(theta)*x||_p - r
                # h = (np.abs(error_1)**p + np.abs(error_2)**p)**(1.0/p) - safety_width/2.0
                h = (np.abs(u)**p + np.abs(v)**p)**(1.0/p) - safety_width/2.0
                # h_square_dot1 = (np.sin(theta[i])*error_2*np.abs(error_2) + np.cos(theta[i])*error_1*np.abs(error_1))/(error_2**2*np.abs(error_2) + error_1**2*np.abs(error_1))**(2/3)
                # h_square_dot2 = (np.sin(theta[i])*error_1*np.abs(error_1) - np.cos(theta[i])*error_2*np.abs(error_2))/(error_1**2*np.abs(error_1) + error_2**2*np.abs(error_2))**(2/3)
                
                # 3-norm “square”
                den = (np.abs(u)**p + np.abs(v)**p)**((p-1)/p)
                uu = u * np.abs(u)
                vv = v * np.abs(v)
                h_square_dot1 = (c*uu + s*vv) / den                            # ∂/∂ex
                h_square_dot2 = (-s*uu + c*vv) / den                           # ∂/∂ey


                A[count, 2*i] = -h_square_dot1
                A[count, 2*i+1] = -h_square_dot2
                A[count, 2*j] = h_square_dot1
                A[count, 2*j+1] = h_square_dot2

                b[count] = barrier_gain*np.power(h, 3)
                count += 1

                if h < h_min:
                    h_min = h

        for i in range(N):
                    xi = x[:, i]
                    for k in range(K):
                        ck = obstacle_centers[:, k]
                        err = xi - ck
                        h_obs = (err[0] * err[0] + err[1] * err[1]) - np.power(obstacle_radius, 2)

                        # Only agent i's control appears
                        A[count, (2 * i, 2 * i + 1)] = -2 * err
                        b[count] = 0.5 * np.power(h_obs, 3) # fixed gain
                        count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

        f = -2*np.reshape(dxi, 2*N, order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F'), h_min

    return f

def create_single_integrator_barrier_certificate_time_varying_with_obstacles(Delta, lamb,target_shape, Delta_dot, safety_radius=0.17, barrier_gain=100, safety_a=0.17, safety_b=0.12, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system with circle to elliptical time varying safety region.
    
    barrier_gain: double (controls how quickly agents can approach each other. lower = slower)
    safety_a: double (semi-major axis of the safety ellipse)
    safety_b: double (semi-minor axis of the safety ellipse)
    magnitude_limit: double (how fast the robot can move linearly)

    -> function (the barrier certificate function)
    """

    # Check user input types
    assert isinstance(barrier_gain, (int, float)), "The barrier gain must be an integer or float. Received type %r." % type(barrier_gain).__name__
    assert isinstance(safety_a, (int, float)), "The semi-major axis of the safety ellipse must be an integer or float. Received type %r." % type(safety_a).__name__
    assert isinstance(safety_b, (int, float)), "The semi-minor axis of the safety ellipse must be an integer or float. Received type %r." % type(safety_b).__name__
    assert isinstance(magnitude_limit, (int, float)), "The maximum linear velocity of the robot must be an integer or float. Received type %r." % type(magnitude_limit).__name__

    # Check user input ranges/sizes
    assert barrier_gain > 0, "The barrier gain must be positive. Received %r." % barrier_gain
    assert safety_a > 0 and safety_b > 0, "The semi-major and semi-minor axes must be positive. Received a: %r, b: %r." % (safety_a, safety_b)
    assert magnitude_limit > 0, "The maximum linear velocity of the robot must be positive. Received %r." % magnitude_limit

    def f(dxi, x,theta):
        # Check user input types
        assert isinstance(dxi, np.ndarray), "The single-integrator robot velocity command must be a numpy array. Received type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "The robot states must be a numpy array. Received type %r." % type(x).__name__

        # Check user input ranges/sizes
        assert x.shape[0] == 2, "The dimension of the single integrator robot states must be 2 ([x;y]). Received dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "The dimension of the robot single integrator velocity command must be 2 ([x_dot;y_dot]). Received dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "The number of robot states must be equal to the number of robot single integrator velocity commands. Received x: %r x %r, dxi: %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        obstacle_centers = np.array([[-0.6,  0.6],
                                 [0,  0]], dtype=float)  # shape (2, 2)
        obstacle_radius = 0.25 # fixed radius
        K = 2 # number of obstacles

        # Initialize variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(N * (N - 1)/2)+N*K
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2 * np.identity(2*N)))

        # Derivative of Delta
        ## Constant function 
        # if Delta < 1:
        #     Delta_dot = 1/T # compute delta dot
        # else:
        #     Delta_dot = 0
        
        count = 0

        # Centralized QP
        for i in range(N-1):
            for j in range(i+1,N):
        # for i in range(N):
        #     for j in range(N):
        #         if i == j:
        #             continue
                error = x[:, i] - x[:, j]
                ###################################### library ########################################
                
                ex, ey = error[0], error[1]
                c = np.cos(theta[i]); s = np.sin(theta[i])

                # --- rotate by +θ: [u;v] = [[c,-s],[s,c]] [ex;ey]
                u =  c*ex - s*ey
                v =  s*ex + c*ey

                # ---------------- h values ----------------
                # circle
                h_circ  = (u/safety_radius)**2 + (v/safety_radius)**2 - 1.0

                # ellipse
                h_ellip = (u/safety_a)**2 + (v/safety_b)**2 - 1.0

                # 3-norm “square”
                safety_width = 0.4  # if not set elsewhere
                h_square = (np.abs(u)**3 + np.abs(v)**3)**(1.0/3.0) - safety_width/2.0

                # triangle (log-sum-exp), numerically stable
                rt3 = np.sqrt(3.0)
                L1 = 4*u + 4*rt3*v
                L2 = -8*u
                L3 = 4*u - 4*rt3*v
                M  = np.maximum.reduce([L1, L2, L3])
                w1 = np.exp(L1 - M); w2 = np.exp(L2 - M); w3 = np.exp(L3 - M)
                S  = w1 + w2 + w3
                h_tri = (3.0/5.0) * (M + np.log(S)) - 1.0

                # convex combo
                h_cur = lamb[0]*h_circ + lamb[1]*h_ellip + lamb[2]*h_tri + lamb[3]*h_square

                # ---------------- gradients wrt (ex, ey) ----------------
                # circle
                h_circ_dot1 = 2.0 * ex / (safety_radius**2)
                h_circ_dot2 = 2.0 * ey / (safety_radius**2)

                # ellipse  (du/dex=c, du/dey=-s, dv/dex=s, dv/dey=c)
                inv_a2 = 1.0 / (safety_a**2)
                inv_b2 = 1.0 / (safety_b**2)
                h_ellip_dot1 = 2.0 * (u * c * inv_a2 + v * s * inv_b2)        # ∂/∂ex
                h_ellip_dot2 = 2.0 * (-u * s * inv_a2 + v * c * inv_b2)       # ∂/∂ey

                # 3-norm “square”
                den = (np.abs(u)**3 + np.abs(v)**3)**(2.0/3.0)
                uu = u * np.abs(u)
                vv = v * np.abs(v)
                h_square_dot1 = (c*uu + s*vv) / den                            # ∂/∂ex
                h_square_dot2 = (-s*uu + c*vv) / den                           # ∂/∂ey

                # triangle: first ∂h/∂u, ∂h/∂v via softmax weights, then chain to (ex,ey)
                a1 = w1 / S; a2 = w2 / S; a3 = w3 / S
                h_u = (3.0/5.0) * (4*a1 - 8*a2 + 4*a3)
                h_v = (3.0/5.0) * (4*rt3*(a1 - a3))
                h_tri_dot1 = c*h_u + s*h_v                                      # ∂/∂ex
                h_tri_dot2 = -s*h_u + c*h_v                                     # ∂/∂ey

                # blended gradient
                h_cur_dot1 = lamb[0]*h_circ_dot1 + lamb[1]*h_ellip_dot1 + lamb[2]*h_tri_dot1 + lamb[3]*h_square_dot1
                h_cur_dot2 = lamb[0]*h_circ_dot2 + lamb[1]*h_ellip_dot2 + lamb[2]*h_tri_dot2 + lamb[3]*h_square_dot2
                #########################################################################################
               
                if target_shape == 1:
                    h_tv = (1-Delta) * h_cur + Delta * h_circ 
                    # for time derivative 
                    diff = h_circ - h_cur

                    A[count, 2*i] = -((1-Delta) * h_cur_dot1 + Delta * h_circ_dot1)
                    A[count, 2*i+1] = -((1-Delta) * h_cur_dot2 + Delta * h_circ_dot2)
                    A[count, 2*j] = (1-Delta) * h_cur_dot1 + Delta * h_circ_dot1
                    A[count, 2*j+1] = (1-Delta) * h_cur_dot2 + Delta * h_circ_dot2
                elif target_shape == 2:
                    h_tv = (1-Delta) * h_cur + Delta * h_ellip 
                    # for time derivative 
                    diff = h_ellip - h_cur

                    A[count, 2*i] = -((1-Delta) * h_cur_dot1 + Delta * h_ellip_dot1)
                    A[count, 2*i+1] = -((1-Delta) * h_cur_dot2 + Delta * h_ellip_dot2)
                    A[count, 2*j] = (1-Delta) * h_cur_dot1 + Delta * h_ellip_dot1
                    A[count, 2*j+1] = (1-Delta) * h_cur_dot2 + Delta * h_ellip_dot2

                elif target_shape == 3:
                    h_tv = (1-Delta) * h_cur + Delta * h_tri 
                    # for time derivative 
                    diff = h_tri - h_cur
                    
                    A[count, 2*i] = -((1-Delta) * h_cur_dot1 + Delta * h_tri_dot1)
                    A[count, 2*i+1] = -((1-Delta) * h_cur_dot2 + Delta * h_tri_dot2)
                    A[count, 2*j] = (1-Delta) * h_cur_dot1 + Delta * h_tri_dot1
                    A[count, 2*j+1] = (1-Delta) * h_cur_dot2 + Delta * h_tri_dot2
                elif target_shape == 4:
                    h_tv = (1-Delta) * h_cur + Delta * h_square 
                    # for time derivative 
                    diff = h_square - h_cur
                    
                    A[count, 2*i] = -((1-Delta) * h_cur_dot1 + Delta * h_square_dot1)
                    A[count, 2*i+1] = -((1-Delta) * h_cur_dot2 + Delta * h_square_dot2)
                    A[count, 2*j] = (1-Delta) * h_cur_dot1 + Delta * h_square_dot1
                    A[count, 2*j+1] = (1-Delta) * h_cur_dot2 + Delta * h_square_dot2
                
                # class k function and time derivative in delta 
                b[count] = barrier_gain * h_tv**3 + Delta_dot *diff
                #  Delta_dot * diff
                count += 1

        for i in range(N):
                    xi = x[:, i]
                    for k in range(K):
                        ck = obstacle_centers[:, k]
                        err = xi - ck
                        h_obs = (err[0] * err[0] + err[1] * err[1]) - np.power(obstacle_radius, 2)

                        # Only agent i's control appears
                        A[count, (2 * i, 2 * i + 1)] = -2 * err
                        b[count] = 0.5 * h_obs # fixed gain
                        count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, axis=0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit / norms[idxs_to_normalize]

        f = -2 * np.reshape(dxi, (2*N,), order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        result = np.reshape(result, (2, -1), order='F')
        
        norms2 = np.linalg.norm(result, 2, axis=0)
        idxs_to_normalize2 = (norms2 > magnitude_limit)
        result[:, idxs_to_normalize2] *= magnitude_limit / norms2[idxs_to_normalize2]

        return result

    return f


def create_single_integrator_barrier_certificate_with_boundary_ellipse(barrier_gain=100, safety_a=0.17, safety_b=0.12, magnitude_limit=0.2, boundary_points = np.array([-1.6, 1.6, -1.2, 1.2])):
    """Creates a barrier certificate for a single-integrator system with an elliptical safety region with a rectangular boundary included.  This function
    returns another function for optimization reasons.

    barrier_gain: double (controls how quickly agents can approach each other. lower = slower)
    safety_a: double (semi-major axis of the safety ellipse)
    safety_b: double (semi-minor axis of the safety ellipse)
    magnitude_limit: double (how fast the robot can move linearly)

    -> function (the barrier certificate function)
    """

   # Check user input types
    assert isinstance(barrier_gain, (int, float)), "The barrier gain must be an integer or float. Received type %r." % type(barrier_gain).__name__
    assert isinstance(safety_a, (int, float)), "The semi-major axis of the safety ellipse must be an integer or float. Received type %r." % type(safety_a).__name__
    assert isinstance(safety_b, (int, float)), "The semi-minor axis of the safety ellipse must be an integer or float. Received type %r." % type(safety_b).__name__
    assert isinstance(magnitude_limit, (int, float)), "The maximum linear velocity of the robot must be an integer or float. Received type %r." % type(magnitude_limit).__name__
    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_a > 0 and safety_b > 0, "The semi-major and semi-minor axes must be positive. Received a: %r, b: %r." % (safety_a, safety_b)
    assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    def f(dxi, x,theta):
        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "In the function created by the create_single_integrator_barrier_certificate function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        
        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(comb(N, 2))*2 + 4*N
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        #H = sparse(matrix(2*np.identity(2*N)))
        H = 2*np.identity(2*N)
        # theta=theta+np.pi/4

        count = 0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                error = x[:, i] - x[:, j]
                error_1 = (error[0]*np.cos(theta[i])+error[1]*np.sin(theta[i])) / safety_a
                error_2 = (error[0]*np.sin(theta[i])-error[1]*np.cos(theta[i])) / safety_b
                h = error_1**2 + error_2**2 - 1    

                # if np.mod(i,2) == 0:
                A[count, 2*i] = -2 * ((error[0])*np.cos(theta[i])+(error[1])*np.sin(theta[i]))*np.cos(theta[i])/ safety_a**2 - 2 * ((error[0])*np.sin(theta[i])-(error[1])*np.cos(theta[i]))*np.sin(theta[i])/ safety_b**2
                A[count, 2*i+1] = -2 * ((error[0])*np.cos(theta[i])+(error[1])*np.sin(theta[i]))*np.sin(theta[i])/ safety_a**2 - 2 * ((error[0])*np.sin(theta[i])-(error[1])*np.cos(theta[i]))*-np.cos(theta[i])/ safety_b**2
                A[count, 2*j] = 2 * ((error[0])*np.cos(theta[i])+(error[1])*np.sin(theta[i]))*np.cos(theta[i])/ safety_a**2 + 2 * ((error[0])*np.sin(theta[i])-(error[1])*np.cos(theta[i]))*np.sin(theta[i])/ safety_b**2
                A[count, 2*j+1] = 2 * ((error[0])*np.cos(theta[i])+(error[1])*np.sin(theta[i]))*np.sin(theta[i])/ safety_a**2 + 2 * ((error[0])*np.sin(theta[i])-(error[1])*np.cos(theta[i]))*-np.cos(theta[i])/ safety_b**2
                # else:
                #     A[count, 2*i] = -2 * ((error[0])*np.cos(theta[i])-(error[1])*np.sin(theta[i]))*np.cos(theta[i])/ safety_b**2 - 2 * ((error[0]-safety_a)*np.sin(theta[i])+(error[1])*np.cos(theta[i]))*np.sin(theta[i])/ safety_a**2
                #     A[count, 2*i+1] = -2 * ((error[0])*np.cos(theta[i])-(error[1])*np.sin(theta[i]))*-np.sin(theta[i])/ safety_b**2 - 2 * ((error[0])*np.sin(theta[i])+(error[1])*np.cos(theta[i]))*np.cos(theta[i])/ safety_a**2
                #     A[count, 2*j] = 2 * ((error[0])*np.cos(theta[i])-(error[1])*np.sin(theta[i]))*np.cos(theta[i])/ safety_b**2 + 2 * ((error[0])*np.sin(theta[i])+(error[1])*np.cos(theta[i]))*np.sin(theta[i])/ safety_a**2
                #     A[count, 2*j+1] = 2 * ((error[0])*np.cos(theta[i])-(error[1])*np.sin(theta[i]))*-np.sin(theta[i])/ safety_b**2 + 2 * ((error[0])*np.sin(theta[i])+(error[1])*np.cos(theta[i]))*np.cos(theta[i])/ safety_a**2

                b[count] = barrier_gain * h**3
                count += 1
        
        for k in range(N):
            # Set it as static circle barrier 
            safety_radius = 0.07 
            #Pos Y
            A[count, (2*k, 2*k+1)] = np.array([0,1])
            b[count] = 0.4*barrier_gain*(boundary_points[3] - safety_radius/2 - x[1,k])**3  # cubic
            # b[count] = 0.4*barrier_gain*(boundary_points[3] - safety_radius/2 - x[1,k])
            count += 1

            #Neg Y
            A[count, (2*k, 2*k+1)] = -np.array([0,1])
            b[count] = 0.4*barrier_gain*(-boundary_points[2] - safety_radius/2 + x[1,k])**3
            # b[count] = 0.4*barrier_gain*(-boundary_points[2] - safety_radius/2 + x[1,k])
            count += 1

            #Pos X
            A[count, (2*k, 2*k+1)] = np.array([1,0])
            b[count] = 0.4*barrier_gain*(boundary_points[1] - safety_radius/2 - x[0,k])**3
            # b[count] = 0.4*barrier_gain*(boundary_points[1] - safety_radius/2 - x[0,k])
            count += 1

            #Neg X
            A[count, (2*k, 2*k+1)] = -np.array([1,0])
            b[count] = 0.4*barrier_gain*(-boundary_points[0] - safety_radius/2 + x[0,k])**3
            # b[count] = 0.4*barrier_gain*(-boundary_points[0] - safety_radius/2 + x[0,k])
            count += 1
        
        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

        f = -2*np.reshape(dxi, (2*N,1), order='F')
        b = np.reshape(b, (count,1), order='F')
        result = qp(matrix(H), matrix(f), matrix(A), matrix(b))['x']
        #result = solver2.solve_qp(H, f, A, b, 0)[0]

        return np.reshape(result, (2, N), order='F')

    return f

# def create_single_integrator_barrier_certificate_double_circle(barrier_gain=100, safety_radius=0.17, magnitude_limit=0.2):
#     """Creates a barrier certificate for a single-integrator system with a rectangular boundary included.  This function
#     returns another function for optimization reasons.

#     barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
#     safety_radius: double (how far apart the agents will stay)
#     magnitude_limit: how fast the robot can move linearly.

#     -> function (the barrier certificate function)
#     """

#     #Check user input types
#     assert isinstance(barrier_gain, (int, float)), "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
#     assert isinstance(safety_radius, (int, float)), "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
#     assert isinstance(magnitude_limit, (int, float)), "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

#     #Check user input ranges/sizes
#     assert barrier_gain > 0, "In the function create_single_integrator_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
#     assert safety_radius >= 0.12, "In the function create_single_integrator_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m) plus the distance to the look ahead point used in the diffeomorphism if that is being used. Recieved %r." % safety_radius
#     assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
#     assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


#     def f(dxi, x):
#         #Check user input types
#         assert isinstance(dxi, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
#         assert isinstance(x, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

#         #Check user input ranges/sizes
#         assert x.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % x.shape[0]
#         assert dxi.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
#         assert x.shape[1] == dxi.shape[1], "In the function created by the create_single_integrator_barrier_certificate function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        
#         # Initialize some variables for computational savings
#         N = dxi.shape[1]
#         num_constraints = int(comb(N, 2)) + 4*N
#         A = np.zeros((num_constraints, 2*N))
#         b = np.zeros(num_constraints)
#         #H = sparse(matrix(2*np.identity(2*N)))
#         H = 2*np.identity(2*N)

#         #first barrier
#         count = 0
#         for i in range(N-1):
#             for j in range(i+1, N):
#                 error = x[:, i] - x[:, j]
#                 h = np.abs(error[0]) + np.abs(error[1]) - safety_radius  # different shape (diamond)

#                 A[count, (2*i, (2*i+1))] = -2*error
#                 A[count, (2*j, (2*j+1))] = 2*error
#                 b[count] = barrier_gain*np.power(h, 3) #    cubic
#                 # b[count] = barrier_gain*h

#                 count += 1
#         #second barrier
#         for i in range(N-1):
#             for j in range(i+1, N):
#                 error = x[:, i] - x[:, j]
#                 h = np.abs(error[0]) + np.abs(error[1]) - safety_radius  # different shape (diamond)

#                 A[count, (2*i, (2*i+1))] = -2*error
#                 A[count, (2*j, (2*j+1))] = 2*error
#                 b[count] = barrier_gain*np.power(h, 3) #    cubic
#                 # b[count] = barrier_gain*h

#             count += 1
        
#         # Threshold control inputs before QP
#         norms = np.linalg.norm(dxi, 2, 0)
#         idxs_to_normalize = (norms > magnitude_limit)
#         dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

#         f = -np.reshape(dxi, (2*N,1), order='F')
#         b = np.reshape(b, (count,1), order='F')
#         result = qp(matrix(H), matrix(f), matrix(A), matrix(b))['x']
#         #result = solver2.solve_qp(H, f, A, b, 0)[0]

#         return np.reshape(result, (2, N), order='F')

#     return f


def create_single_integrator_barrier_certificate2(barrier_gain=100, unsafe_barrier_gain=1e6, safety_radius=0.17, magnitude_limit=0.2):
    """Creates a barrier certificate for a single-integrator system.  This function
    returns another function for optimization reasons. This function is different from 
    create_single_integrator_barrier_certificate as it changes the barrier gain to a large
    number if the single integrator point enters the unsafe region.

    barrier_gain: double (controls how quickly agents can approach each other.  lower = slower)
    safety_radius: double (how far apart the agents will stay)
    magnitude_limit: how fast the robot can move linearly.

    -> function (the barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_single_integrator_barrier_certificate2, the barrier gain inside the safe set (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(unsafe_barrier_gain, (int, float)), "In the function create_single_integrator_barrier_certificate2, the barrier gain if outside the safe set (unsafe_barrier_gain) must be an integer or float. Recieved type %r." % type(unsafe_barrier_gain).__name__
    assert isinstance(safety_radius, (int, float)), "In the function create_single_integrator_barrier_certificate2, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_single_integrator_barrier_certificate2, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_single_integrator_barrier_certificate2, the barrier gain inside the safe set (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert unsafe_barrier_gain > 0, "In the function create_single_integrator_barrier_certificate2, the barrier gain if outside the safe set (unsafe_barrier_gain) must be positive. Recieved %r." % unsafe_barrier_gain
    assert safety_radius >= 0.12, "In the function create_single_integrator_barrier_certificate2, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m) plus the distance to the look ahead point used in the diffeomorphism if that is being used. Recieved %r." % safety_radius
    assert magnitude_limit > 0, "In the function create_single_integrator_barrier_certificate2, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_single_integrator_barrier_certificate2, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    def f(dxi, x):
        #Check user input types
        assert isinstance(dxi, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate2 function, the single-integrator robot velocity command (dxi) must be a numpy array. Recieved type %r." % type(dxi).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_single_integrator_barrier_certificate2 function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate2 function, the dimension of the single integrator robot states (x) must be 2 ([x;y]). Recieved dimension %r." % x.shape[0]
        assert dxi.shape[0] == 2, "In the function created by the create_single_integrator_barrier_certificate2 function, the dimension of the robot single integrator velocity command (dxi) must be 2 ([x_dot;y_dot]). Recieved dimension %r." % dxi.shape[0]
        assert x.shape[1] == dxi.shape[1], "In the function created by the create_single_integrator_barrier_certificate2 function, the number of robot states (x) must be equal to the number of robot single integrator velocity commands (dxi). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxi.shape[0], dxi.shape[1])

        
        # Initialize some variables for computational savings
        N = dxi.shape[1]
        num_constraints = int(comb(N, 2))*2
        A = np.zeros((num_constraints, 2*N))
        b = np.zeros(num_constraints)
        H = sparse(matrix(2*np.identity(2*N)))

        count = 0
        for i in range(N-1):
            for j in range(i+1, N):
                error = x[:, i] - x[:, j]
                h = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)

                A[count, (2*i, (2*i+1))] = -2*error
                A[count, (2*j, (2*j+1))] = 2*error
                if h >= 0:
                    b[count] = barrier_gain*np.power(h, 3)
                else:
                    b[count] = unsafe_barrier_gain*np.power(h, 3)

                count += 1

        # Threshold control inputs before QP
        norms = np.linalg.norm(dxi, 2, 0)
        idxs_to_normalize = (norms > magnitude_limit)
        dxi[:, idxs_to_normalize] *= magnitude_limit/norms[idxs_to_normalize]

        f = -2*np.reshape(dxi, 2*N, order='F')
        result = qp(H, matrix(f), matrix(A), matrix(b))['x']

        return np.reshape(result, (2, -1), order='F')

    return f

def create_unicycle_barrier_certificate(barrier_gain=100, safety_radius=0.12, projection_distance=0.05, magnitude_limit=0.2):
    """ Creates a unicycle barrier cetifcate to avoid collisions. Uses the diffeomorphism mapping
    and single integrator implementation. For optimization purposes, this function returns 
    another function.

    barrier_gain: double (how fast the robots can approach each other)
    safety_radius: double (how far apart the robots should stay)
    projection_distance: double (how far ahead to place the bubble)

    -> function (the unicycle barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_radius, (int, float)), "In the function create_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
    assert isinstance(projection_distance, (int, float)), "In the function create_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be an integer or float. Recieved type %r." % type(projection_distance).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_radius >= 0.12, "In the function create_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m). Recieved %r." % safety_radius
    assert projection_distance > 0, "In the function create_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be positive. Recieved %r." % projection_distance
    assert magnitude_limit > 0, "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    si_barrier_cert = create_single_integrator_barrier_certificate(barrier_gain=barrier_gain, safety_radius=safety_radius+projection_distance)

    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(projection_distance=projection_distance)

    uni_to_si_dyn = create_uni_to_si_dynamics(projection_distance=projection_distance)

    def f(dxu, x):
        #Check user input types
        assert isinstance(dxu, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the unicycle robot velocity command (dxu) must be a numpy array. Recieved type %r." % type(dxu).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 3, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the unicycle robot states (x) must be 3 ([x;y;theta]). Recieved dimension %r." % x.shape[0]
        assert dxu.shape[0] == 2, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the robot unicycle velocity command (dxu) must be 2 ([v;w]). Recieved dimension %r." % dxu.shape[0]
        assert x.shape[1] == dxu.shape[1], "In the function created by the create_unicycle_barrier_certificate function, the number of robot states (x) must be equal to the number of robot unicycle velocity commands (dxu). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxu.shape[0], dxu.shape[1])


        x_si = uni_to_si_states(x)
        #Convert unicycle control command to single integrator one
        dxi = uni_to_si_dyn(dxu, x)
        #Apply single integrator barrier certificate
        dxi = si_barrier_cert(dxi, x_si)
        #Return safe unicycle command
        return si_to_uni_dyn(dxi, x)

    return f

def create_unicycle_barrier_certificate_diamond(barrier_gain=100, safety_radius=0.12, projection_distance=0.05, magnitude_limit=0.2):
    """ Creates a unicycle barrier cetifcate to avoid collisions. Uses the diffeomorphism mapping
    and single integrator implementation. For optimization purposes, this function returns 
    another function.

    barrier_gain: double (how fast the robots can approach each other)
    safety_radius: double (how far apart the robots should stay)
    projection_distance: double (how far ahead to place the bubble)

    -> function (the unicycle barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_radius, (int, float)), "In the function create_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
    assert isinstance(projection_distance, (int, float)), "In the function create_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be an integer or float. Recieved type %r." % type(projection_distance).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_radius >= 0.12, "In the function create_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m). Recieved %r." % safety_radius
    assert projection_distance > 0, "In the function create_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be positive. Recieved %r." % projection_distance
    assert magnitude_limit > 0, "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    si_barrier_cert = create_single_integrator_barrier_certificate_diamond(barrier_gain=barrier_gain, safety_radius=safety_radius+projection_distance)

    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(projection_distance=projection_distance)

    uni_to_si_dyn = create_uni_to_si_dynamics(projection_distance=projection_distance)

    def f(dxu, x):
        #Check user input types
        assert isinstance(dxu, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the unicycle robot velocity command (dxu) must be a numpy array. Recieved type %r." % type(dxu).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 3, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the unicycle robot states (x) must be 3 ([x;y;theta]). Recieved dimension %r." % x.shape[0]
        assert dxu.shape[0] == 2, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the robot unicycle velocity command (dxu) must be 2 ([v;w]). Recieved dimension %r." % dxu.shape[0]
        assert x.shape[1] == dxu.shape[1], "In the function created by the create_unicycle_barrier_certificate function, the number of robot states (x) must be equal to the number of robot unicycle velocity commands (dxu). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxu.shape[0], dxu.shape[1])


        x_si = uni_to_si_states(x)
        #Convert unicycle control command to single integrator one
        dxi = uni_to_si_dyn(dxu, x)
        #Apply single integrator barrier certificate
        dxi = si_barrier_cert(dxi, x_si)
        #Return safe unicycle command
        return si_to_uni_dyn(dxi, x)

    return f

def create_unicycle_barrier_certificate_ellipse(barrier_gain=100, safety_a=0.17, safety_b=0.12, projection_distance=0.05, magnitude_limit=0.2):
    """ Creates a unicycle barrier cetifcate to avoid collisions. Uses the diffeomorphism mapping
    and single integrator implementation. For optimization purposes, this function returns 
    another function.

    barrier_gain: double (how fast the robots can approach each other)
    safety_radius: double (how far apart the robots should stay)
    projection_distance: double (how far ahead to place the bubble)

    -> function (the unicycle barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_a, (int, float)), "In the function create_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_a).__name__
    assert isinstance(safety_b, (int, float)), "In the function create_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_b).__name__
    assert isinstance(projection_distance, (int, float)), "In the function create_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be an integer or float. Recieved type %r." % type(projection_distance).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_a >= 0.12, "In the function create_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m). Recieved %r." % safety_a
    assert safety_b >= 0.12, "In the function create_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m). Recieved %r." % safety_b
    assert projection_distance > 0, "In the function create_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be positive. Recieved %r." % projection_distance
    assert magnitude_limit > 0, "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    si_barrier_cert = create_single_integrator_barrier_certificate_ellipse(barrier_gain=barrier_gain, safety_a=safety_a+projection_distance,safety_b=safety_b+projection_distance)

    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(projection_distance=projection_distance)

    uni_to_si_dyn = create_uni_to_si_dynamics(projection_distance=projection_distance)

    def f(dxu, x):
        #Check user input types
        assert isinstance(dxu, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the unicycle robot velocity command (dxu) must be a numpy array. Recieved type %r." % type(dxu).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 3, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the unicycle robot states (x) must be 3 ([x;y;theta]). Recieved dimension %r." % x.shape[0]
        assert dxu.shape[0] == 2, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the robot unicycle velocity command (dxu) must be 2 ([v;w]). Recieved dimension %r." % dxu.shape[0]
        assert x.shape[1] == dxu.shape[1], "In the function created by the create_unicycle_barrier_certificate function, the number of robot states (x) must be equal to the number of robot unicycle velocity commands (dxu). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxu.shape[0], dxu.shape[1])


        x_si = uni_to_si_states(x)
        #Convert unicycle control command to single integrator one
        dxi = uni_to_si_dyn(dxu, x)
        #Apply single integrator barrier certificate
        dxi = si_barrier_cert(dxi, x_si)
        #Return safe unicycle command
        return si_to_uni_dyn(dxi, x)

    return f

def create_unicycle_barrier_certificate_with_boundary(barrier_gain=100, safety_radius=0.12, projection_distance=0.05, magnitude_limit=0.2, boundary_points = np.array([-1.6, 1.6, -1.0, 1.0])):
    """ Creates a unicycle barrier cetifcate to avoid collisions. Uses the diffeomorphism mapping
    and single integrator implementation. For optimization purposes, this function returns 
    another function.

    barrier_gain: double (how fast the robots can approach each other)
    safety_radius: double (how far apart the robots should stay)
    projection_distance: double (how far ahead to place the bubble)

    -> function (the unicycle barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(safety_radius, (int, float)), "In the function create_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
    assert isinstance(projection_distance, (int, float)), "In the function create_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be an integer or float. Recieved type %r." % type(projection_distance).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_unicycle_barrier_certificate, the barrier gain (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert safety_radius >= 0.12, "In the function create_unicycle_barrier_certificate, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m). Recieved %r." % safety_radius
    assert projection_distance > 0, "In the function create_unicycle_barrier_certificate, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be positive. Recieved %r." % projection_distance
    assert magnitude_limit > 0, "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_unicycle_barrier_certificate, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary(barrier_gain=barrier_gain, safety_radius=safety_radius+projection_distance, boundary_points=boundary_points)

    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(projection_distance=projection_distance)

    uni_to_si_dyn = create_uni_to_si_dynamics(projection_distance=projection_distance)

    def f(dxu, x):
        #Check user input types
        assert isinstance(dxu, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the unicycle robot velocity command (dxu) must be a numpy array. Recieved type %r." % type(dxu).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 3, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the unicycle robot states (x) must be 3 ([x;y;theta]). Recieved dimension %r." % x.shape[0]
        assert dxu.shape[0] == 2, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the robot unicycle velocity command (dxu) must be 2 ([v;w]). Recieved dimension %r." % dxu.shape[0]
        assert x.shape[1] == dxu.shape[1], "In the function created by the create_unicycle_barrier_certificate function, the number of robot states (x) must be equal to the number of robot unicycle velocity commands (dxu). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxu.shape[0], dxu.shape[1])


        x_si = uni_to_si_states(x)
        #Convert unicycle control command to single integrator one
        dxi = uni_to_si_dyn(dxu, x)
        #Apply single integrator barrier certificate
        dxi = si_barrier_cert(dxi, x_si)
        #Return safe unicycle command
        return si_to_uni_dyn(dxi, x)

    return f

def create_unicycle_barrier_certificate2(barrier_gain=500, unsafe_barrier_gain=1e6, safety_radius=0.12, projection_distance=0.05, magnitude_limit=0.2):
    """ Creates a unicycle barrier cetifcate to avoid collisions. Uses the diffeomorphism mapping
    and single integrator implementation. For optimization purposes, this function returns 
    another function.

    barrier_gain: double (how fast the robots can approach each other)
    safety_radius: double (how far apart the robots should stay)
    projection_distance: double (how far ahead to place the bubble)

    -> function (the unicycle barrier certificate function)
    """

    #Check user input types
    assert isinstance(barrier_gain, (int, float)), "In the function create_unicycle_barrier_certificate2, the barrier gain inside the safe set (barrier_gain) must be an integer or float. Recieved type %r." % type(barrier_gain).__name__
    assert isinstance(unsafe_barrier_gain, (int, float)), "In the function create_unicycle_barrier_certificate2, the barrier gain outside the safe set (unsafe_barrier_gain) must be an integer or float. Recieved type %r." % type(unsafe_barrier_gain).__name__
    assert isinstance(safety_radius, (int, float)), "In the function create_unicycle_barrier_certificate2, the safe distance between robots (safety_radius) must be an integer or float. Recieved type %r." % type(safety_radius).__name__
    assert isinstance(projection_distance, (int, float)), "In the function create_unicycle_barrier_certificate2, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be an integer or float. Recieved type %r." % type(projection_distance).__name__
    assert isinstance(magnitude_limit, (int, float)), "In the function create_unicycle_barrier_certificate2, the maximum linear velocity of the robot (magnitude_limit) must be an integer or float. Recieved type %r." % type(magnitude_limit).__name__

    #Check user input ranges/sizes
    assert barrier_gain > 0, "In the function create_unicycle_barrier_certificate2, the barrier gain inside the safe set (barrier_gain) must be positive. Recieved %r." % barrier_gain
    assert unsafe_barrier_gain > 0, "In the function create_unicycle_barrier_certificate2, the barrier gain outside the safe set (unsafe_barrier_gain) must be positive. Recieved %r." % unsafe_barrier_gain
    assert safety_radius >= 0.12, "In the function create_unicycle_barrier_certificate2, the safe distance between robots (safety_radius) must be greater than or equal to the diameter of the robot (0.12m). Recieved %r." % safety_radius
    assert projection_distance > 0, "In the function create_unicycle_barrier_certificate2, the projected point distance for the diffeomorphism between sinlge integrator and unicycle (projection_distance) must be positive. Recieved %r." % projection_distance
    assert magnitude_limit > 0, "In the function create_unicycle_barrier_certificate2, the maximum linear velocity of the robot (magnitude_limit) must be positive. Recieved %r." % magnitude_limit
    assert magnitude_limit <= 0.2, "In the function create_unicycle_barrier_certificate2, the maximum linear velocity of the robot (magnitude_limit) must be less than the max speed of the robot (0.2m/s). Recieved %r." % magnitude_limit


    si_barrier_cert = create_single_integrator_barrier_certificate2(barrier_gain=barrier_gain, unsafe_barrier_gain=unsafe_barrier_gain, safety_radius=safety_radius+projection_distance)

    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping(projection_distance=projection_distance)

    uni_to_si_dyn = create_uni_to_si_dynamics(projection_distance=projection_distance)

    def f(dxu, x):
        #Check user input types
        assert isinstance(dxu, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the unicycle robot velocity command (dxu) must be a numpy array. Recieved type %r." % type(dxu).__name__
        assert isinstance(x, np.ndarray), "In the function created by the create_unicycle_barrier_certificate function, the robot states (x) must be a numpy array. Recieved type %r." % type(x).__name__

        #Check user input ranges/sizes
        assert x.shape[0] == 3, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the unicycle robot states (x) must be 3 ([x;y;theta]). Recieved dimension %r." % x.shape[0]
        assert dxu.shape[0] == 2, "In the function created by the create_unicycle_barrier_certificate function, the dimension of the robot unicycle velocity command (dxu) must be 2 ([v;w]). Recieved dimension %r." % dxu.shape[0]
        assert x.shape[1] == dxu.shape[1], "In the function created by the create_unicycle_barrier_certificate function, the number of robot states (x) must be equal to the number of robot unicycle velocity commands (dxu). Recieved a current robot pose input array (x) of size %r x %r and single integrator velocity array (dxi) of size %r x %r." % (x.shape[0], x.shape[1], dxu.shape[0], dxu.shape[1])


        x_si = uni_to_si_states(x)
        #Convert unicycle control command to single integrator one
        dxi = uni_to_si_dyn(dxu, x)
        #Apply single integrator barrier certificate
        dxi = si_barrier_cert(dxi, x_si)
        #Return safe unicycle command
        return si_to_uni_dyn(dxi, x)

    return f

def create_unicycle_differential_drive_barrier_certificate(max_num_obstacle_points = 100, max_num_robots = 30, disturbance = 5, wheel_vel_limit = 12.5, base_length = 0.105, wheel_radius = 0.016,
    projection_distance =0.05, barrier_gain = 150, safety_radius = 0.17):
    

    D = np.matrix([[wheel_radius/2, wheel_radius/2], [-wheel_radius/base_length, wheel_radius/base_length]])
    L = np.matrix([[1,0],[0,projection_distance]])* D
    disturb = np.matrix([[-disturbance, -disturbance, disturbance, disturbance],[-disturbance, disturbance, disturbance, -disturbance]])
    num_disturbs = np.size(disturb[1,:])

    max_num_constraints = (max_num_robots**2-max_num_robots)//2 + max_num_robots*max_num_obstacle_points
    A = np.matrix(np.zeros([max_num_constraints, 2*max_num_robots]))
    b = np.matrix(np.zeros([max_num_constraints, 1]))
    Os = np.matrix(np.zeros([2,max_num_robots]))
    ps = np.matrix(np.zeros([2,max_num_robots]))
    Ms = np.matrix(np.zeros([2,2*max_num_robots]))

    def robust_barriers(dxu, x, obstacles=np.empty(0)):

        num_robots = np.size(dxu[0,:])

        if obstacles.size != 0:
            num_obstacles = np.size(obstacles[0,:])
        else:
            num_obstacles = 0

        if(num_robots < 2):
            temp = 0
        else:
            temp = (num_robots**2-num_robots)//2


        # Generate constraints for barrier certificates based on the size of the safety radius
        num_constraints = temp + num_robots*num_obstacles
        A[0:num_constraints, 0:2*num_robots] = 0
        Os[0, 0:num_robots] = np.cos(x[2, :])
        Os[1, 0:num_robots] = np.sin(x[2, :])
        ps[:, 0:num_robots] = x[0:2, :] + projection_distance*Os[:, 0:num_robots]
        # Ms Real Form
        # Ms[0, 0:2*num_robots:2] = Os[0, 0:num_robots]
        # Ms[0, 1:2*num_robots:2] = -projection_distance*Os[1, 0:num_robots]
        # Ms[1, 1:2*num_robots:2] = projection_distance*Os[0, 0:num_robots]
        # Ms[1, 0:2*num_robots:2] = Os[1, 0:num_robots]
        # Flipped Ms to be able to perform desired matrix multiplication
        Ms[0, 0:2*num_robots:2] = Os[0, 0:num_robots]
        Ms[0, 1:2*num_robots:2] = Os[1, 0:num_robots]
        Ms[1, 1:2*num_robots:2] = projection_distance*Os[0, 0:num_robots]
        Ms[1, 0:2*num_robots:2] = -projection_distance*Os[1, 0:num_robots]
        MDs  = (Ms.T * D).T
        temp = np.copy(MDs[1, 0:2*num_robots:2])
        MDs[1, 0:2*num_robots:2] =  MDs[0, 1:2*num_robots:2]
        MDs[0, 1:2*num_robots:2] = temp

        count = 0

        for i in range(num_robots-1):
            diffs = ps[:,i] - ps[:, i+1:num_robots]
            hs = np.sum(np.square(diffs),0) - safety_radius**2 # 1 by N
            h_dot_is = 2*diffs.T*MDs[:,(2*i, 2*i+1)] # N by 2
            h_dot_js = np.matrix(np.zeros((2,num_robots - (i+1))))
            h_dot_js[0, :] = -np.sum(2*np.multiply(diffs, MDs[:,2*(i+1):2*num_robots:2]), 0)
            h_dot_js[1, :] = -np.sum(2*np.multiply(diffs, MDs[:,2*(i+1)+1:2*num_robots:2]), 0)
            new_constraints = num_robots - i - 1
            A[count:count+new_constraints, (2*i):(2*i+2)] = h_dot_is
            A[range(count,count+new_constraints), range(2*(i+1),2*num_robots,2)] = h_dot_js[0,:]
            A[range(count,count+new_constraints), range(2*(i+1)+1,2*num_robots,2)] = h_dot_js[1,:]
            b[count:count+new_constraints] = -barrier_gain*(np.power(hs,3)).T - np.min(h_dot_is*disturb,1) - np.min(h_dot_js.T*disturb,1)
            count += new_constraints

        if obstacles.size != 0:
            # Do obstacles
            for i in range(num_robots):
                diffs = (ps[:, i] - obstacles)
                h = np.sum(np.square(diffs),0) - safety_radius**2
                h_dot_i = 2*diffs.T*MDs[:,2*i:2*i+2]
                A[count:count+num_obstacles,(2*i):(2*i+2)] = h_dot_i
                b[count:count+num_obstacles] = -barrier_gain*(np.power(h,3)).T  - np.min(h_dot_i*disturb, 1)
                count = count + num_obstacles

        # Adding Upper Bounds On Wheels
        A[count:count+2*num_robots,0:2*num_robots] = -np.eye(2*num_robots)
        b[count:count+2*num_robots] = -wheel_vel_limit
        count += 2*num_robots
        # # Adding Lower Bounds on Wheels
        A[count:count+2*num_robots,0:2*num_robots] = np.eye(2*num_robots)
        b[count:count+2*num_robots] = -wheel_vel_limit
        count += 2*num_robots

        # Solve QP program generated earlier
        L_all = np.kron(np.eye(num_robots), L)
        dxu = np.linalg.inv(D)*dxu # Convert user input to differential drive
        vhat = np.matrix(np.reshape(dxu ,(2*num_robots,1), order='F'))
        H = 2*L_all.T*L_all
        f = np.transpose(-2*np.transpose(vhat)*np.transpose(L_all)*L_all)

        # Alternative Solver
        #start = time.time()
        #vnew2 = solvers.qp(matrix(H), matrix(f), -matrix(A[0:count,0:2*num_robots]), -matrix( b[0:count]))['x'] # , A, b) Omit last 2 arguments since our QP has no equality constraints
        #print("Time Taken by cvxOpt: {} s".format(time.time() - start))

        vnew = solver2.solve_qp(H, -np.squeeze(np.array(f)), A[0:count,0:2*num_robots].T, np.squeeze(np.array(b[0:count])))[0]
        # Initial Guess for Solver at the Next Iteration
        # vnew = quadprog(H, double(f), -A(1:num_constraints,1:2*num_robots), -b(1:num_constraints), [], [], -wheel_vel_limit*ones(2*num_robots,1), wheel_vel_limit*ones(2*num_robots,1), [], opts);
        # Set robot velocities to new velocities
        dxu = np.reshape(vnew, (2, num_robots), order='F')
        dxu = D*dxu

        return dxu

    return robust_barriers

def create_unicycle_differential_drive_barrier_certificate_with_boundary(max_num_obstacle_points = 100, max_num_robots = 30, disturbance = 5, wheel_vel_limit = 12.5, base_length = 0.105, wheel_radius = 0.016,
    projection_distance =0.05, barrier_gain = 150, safety_radius = 0.17, boundary_points = np.array([-1.6, 1.6, -1.0, 1.0])):
    

    D = np.array([[wheel_radius/2, wheel_radius/2], [-wheel_radius/base_length, wheel_radius/base_length]])
    L = np.array([[1,0],[0,projection_distance]]).dot(D)
    disturb = np.array([[-disturbance, -disturbance, disturbance, disturbance],[-disturbance, disturbance, disturbance, -disturbance]])
    num_disturbs = disturb.shape[1]

    max_num_constraints = (max_num_robots**2-max_num_robots)//2 + max_num_robots*max_num_obstacle_points
    A = np.zeros([max_num_constraints, 2*max_num_robots])
    b = np.zeros([max_num_constraints, 1])
    Os = np.zeros([2,max_num_robots])
    ps = np.zeros([2,max_num_robots])
    Ms = np.zeros([2,2*max_num_robots])

    def robust_barriers(dxu, x, obstacles=np.empty(0)):

        num_robots = np.size(dxu[0,:])

        if obstacles.size != 0:
            num_obstacles = np.size(obstacles[0,:])
        else:
            num_obstacles = 0

        if(num_robots < 2):
            temp = 0
        else:
            temp = (num_robots**2-num_robots)//2


        # Generate constraints for barrier certificates based on the size of the safety radius
        num_constraints = temp + num_robots*num_obstacles + 4*num_robots
        A[0:num_constraints, 0:2*num_robots] = 0
        Os[0, 0:num_robots] = np.cos(x[2, :])
        Os[1, 0:num_robots] = np.sin(x[2, :])
        ps[:, 0:num_robots] = x[:2, :] + projection_distance*Os[:, 0:num_robots]
        Ms[0, 0:2*num_robots:2] = Os[0, 0:num_robots]
        Ms[0, 1:2*num_robots:2] = -projection_distance*Os[1, 0:num_robots]
        Ms[1, 1:2*num_robots:2] = projection_distance*Os[0, 0:num_robots]
        Ms[1, 0:2*num_robots:2] = Os[1, 0:num_robots]
        ret = np.zeros([1,temp])

        count = 0

        for i in range(num_robots-1):
            for j in range(i+1, num_robots):
                diff = ps[:, [i]] - ps[:, [j]]
                h = np.sum(np.square(diff),0) - safety_radius**2
                h_dot_i = 2*diff.T.dot(Ms[:, ((2*i), (2*i+1))].dot(D))
                h_dot_j = -2*diff.T.dot(Ms[:, ((2*j), (2*j+1))].dot(D))
                h_dot_i = np.reshape(h_dot_i, (1,2))
                h_dot_j = np.reshape(h_dot_j, (1,2))
                A[count, ((2*i), (2*i+1))]=h_dot_i
                A[count, ((2*j), (2*j+1))]=h_dot_j
                b[count] = -barrier_gain*(np.power(h,3))  - np.min(h_dot_i.dot(disturb), 1) - np.min(h_dot_j.dot(disturb), 1)
                count += 1

        if obstacles.size != 0:
            # Do obstacles
            for i in range(num_robots):
                diffs = (ps[:, i] - obstacles)
                h = np.sum(np.square(diff),0) - safety_radius**2
                h_dot_i = 2*diffs*Ms[:, (2*i, 2*i+1)].dot(D)
                A[count:count+num_obstacles, ((2*i),(2*i+1))] = h_dot_i
                b[count:count+num_obstacles] = -barrier_gain*(np.power(h,3))  - np.min(h_dot_i.dot(disturb), 1)
                count = count + num_obstacles

        for k in range(num_robots):
            #Pos Y
            A[count, (2*k, 2*k+1)] = -Ms[1,(2*k,2*k+1)].dot(D)
            b[count] = -0.4*barrier_gain*(boundary_points[3] - safety_radius/2 - ps[1,k])**3;
            count += 1

            #Neg Y
            A[count, (2*k, 2*k+1)] = Ms[1,(2*k,2*k+1)].dot(D)
            b[count] = -0.4*barrier_gain*(-boundary_points[2] - safety_radius/2 + ps[1,k])**3;
            count += 1

            #Pos X
            A[count, (2*k, 2*k+1)] = -Ms[0,(2*k,2*k+1)].dot(D)
            b[count] = -0.4*barrier_gain*(boundary_points[1] - safety_radius/2 - ps[0,k])**3;
            count += 1

            #Neg X
            A[count, (2*k, 2*k+1)] = Ms[0,(2*k,2*k+1)].dot(D)
            b[count] = -0.4*barrier_gain*(-boundary_points[0] - safety_radius/2 + ps[0,k])**3;
            count += 1

        # Adding Upper Bounds On Wheels
        A[count:count+2*num_robots,0:2*num_robots] = -np.eye(2*num_robots)
        b[count:count+2*num_robots] = -wheel_vel_limit
        count += 2*num_robots
        # # Adding Lower Bounds on Wheels
        A[count:count+2*num_robots,0:2*num_robots] = np.eye(2*num_robots)
        b[count:count+2*num_robots] = -wheel_vel_limit
        count += 2*num_robots

        # Solve QP program generated earlier
        L_all = np.kron(np.eye(num_robots), L)
        dxu = np.linalg.inv(D).dot(dxu) # Convert user input to differential drive
        vhat = np.reshape(dxu ,(2*num_robots,1), order='F')
        H = 2*L_all.T.dot(L_all)
        f = -2*vhat.T.dot(L_all.T.dot(L_all))

        # Alternative Solver
        #start = time.time()
        vnew = qp(matrix(H), matrix(f.T), -matrix(A[0:count,0:2*num_robots]), -matrix( b[0:count]))['x'] # , A, b) Omit last 2 arguments since our QP has no equality constraints
        #print("Time Taken by cvxOpt: {} s".format(time.time() - start))

        # vnew = solver2.solve_qp(H, np.float64(f), -A[0:count,0:2*num_robots], -np.array(b[0:count]))[0]
        # Initial Guess for Solver at the Next Iteration
        # vnew = quadprog(H, double(f), -A(1:num_constraints,1:2*num_robots), -b(1:num_constraints), [], [], -wheel_vel_limit*ones(2*num_robots,1), wheel_vel_limit*ones(2*num_robots,1), [], opts);
        # Set robot velocities to new velocities
        dxu = np.reshape(vnew, (2, -1), order='F')
        dxu = D.dot(dxu)

        return dxu

    return robust_barriers

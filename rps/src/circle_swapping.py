import rps.robotarium as robotarium
from rps.utilities.transformations import *
from barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import time

'''This code is for the multi-robot circluar swapping simulations/experiments'''

# The robots will never reach their goal points so set iteration number (not used here, but if deadlock happens we can make the program stop in short time)
iterations = 600

## The arena is bounded between x \in (-1.6,1.6)  y\in (-1,1) 

# Number of robots
N = 16 # 2,4,8,11,16,20

# radius of the circle robots are forming
circle_radius = 0.9

# Calculate initial positions in a circular formation
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)  # Angles for each robot
initial_x = circle_radius * np.cos(theta)  # X coordinates
initial_y = circle_radius * np.sin(theta)  # Y coordinates

# Headings (facing inward)
initial_heading = theta + np.pi  # Heading towards the center (add pi to point inward)

# Combine initial positions into the required format (x, y, theta)
initial_conditions = np.array([initial_x, initial_y, initial_heading])
# Instantiate Robotarium object
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True, initial_conditions=initial_conditions)
# Define goal points: for a swapping behavior, we can simply offset the current positions
goal_points = np.array([initial_x, initial_y, theta])  # Start by setting goal points to the current positions
goal_points[0, :] = -initial_conditions[0,:] 
goal_points[1, :] = -initial_conditions[1,:]

## Plotting Parameters

#GT color scheme
# Gold = np.array([179,163,105])/255
# Navy = np.array([0,48,87])/255
# piMile= np.array([214,219,212])/255
# Black = np.array([0,0,0])
# CM = np.vstack([Gold,Navy,piMile,Black])


# CM = np.random.rand(N,3) # Random Colors

# Use a predefined colormap from Matplotlib
cmap = plt.get_cmap("tab20")  # You can try "Set3", "tab10", or any other colormap

# Generate colors from the colormap for N agents
CM = cmap(np.linspace(0, 1, N))  # Generate N colors evenly spaced within the colormap

##################################### For Robotarium##################################
# Set CM to be all black for N agents (better visibility)
# CM = np.array([[0, 0, 0]] * N)
 
# Default Barrier Parameters (for visualization) 
safety_radius_view = 0.15
safety_radius_marker_size = determine_marker_size(r,safety_radius_view) # Will scale the plotted markers to be the diameter of provided argument (in meters)
font_height_meters = 0.2
font_height_points = determine_font_size(r,font_height_meters) # Will scale the plotted font height to that of the provided argument (in meters)

# Create single integrator position controller
si_position_controller = create_si_position_controller()

# Initialize parameters
radius = 0.25
a = 0.25
b = 0.20

############################################ CBF Library #######################################################
# We're working in single-integrator dynamics, and we don't want the robots
# to collide.  Thus, we're going to use barrier certificates (in a centrialized way)
CBF_n = 2 # how many CBFs we are using 
si_barrier_cert_cir = create_single_integrator_barrier_certificate(barrier_gain=1,safety_radius=radius)
si_barrier_cert_ellip = create_single_integrator_barrier_certificate_ellipse(barrier_gain=1,safety_a=a,safety_b=b)

############################# Pre construction for time-varying CBFs ###########################################
# W1 = np.linspace(1.0, 0.0, 11)
# W  = np.column_stack([W1, 1.0 - W1])  # shape (11, 2)

# def closest_weight_index(Delta):
#     """Return idx ∈ [0..10] for the closest (w1,w2) on 0.1 grid, enforcing x+y=1."""
#     d = np.asarray(Delta, dtype=float)
#     s = d.sum()
#     if s != 0.0:
#         d = d / s                  # normalize minor drift
#     d = np.clip(d, 0.0, 1.0)
#     # index 0 → w1=1.0; index 10 → w1=0.0
#     return int(np.clip(np.round((1.0 - d[0]) * 10.0), 0, 10))

# # Discretize CBFs to create a list with a resolution of 0.1
# def build_certs_for_shape(target_shape, *, barrier_gain, radius, a, b):
#     """Precreate 11 certificate functions for one shape."""
#     certs = []
#     for w1 in W1:
#         cert = create_single_integrator_barrier_certificate_time_varying(
#             [float(w1), 1.0 - float(w1)], T=1, eta= np.sqrt(2)/1*1/3,
#             target_shape=target_shape,
#             barrier_gain=barrier_gain,
#             safety_radius=radius,
#             safety_a=a, safety_b=b
#         )
#         certs.append(cert)
#     return certs

# # Precompute once (so the loop is just an index + lookup)
# certs_by_shape = {
#     1: build_certs_for_shape(1, barrier_gain=1, radius=radius, a=a, b=b),
#     2: build_certs_for_shape(2, barrier_gain=1, radius=radius, a=a, b=b),
# }

# def closest_weight_index(Delta):
#     d = np.asarray(Delta, dtype=float)
#     s = d.sum()
#     if s != 0.0:
#         d = d / s  # normalize if there's tiny drift
#     d = np.clip(d, 0.0, 1.0)
#     # Because candidates are multiples of 0.1 on x, nearest is just rounding x to tenths:
#     # Map x ∈ [0,1] to index 0..10 where index 0 → w1=1.0, index 10 → w1=0.0
#     idx = int(np.floor((1.0 - d[0]) * 10.0 + 0.5))    # robust "round to nearest"
#     idx = int(np.clip(idx, 0, 10))
#     return idx

# def pick_cert_for_Delta(Delta, target_shape):
#     """
#     Returns:
#       cert  : the precomputed barrier certificate function
#       idx   : the chosen index
#       wpair : the chosen (w1, w2)
#     """
#     idx = closest_weight_index(Delta)
#     cert_list = certs_by_shape[target_shape]
#     print(idx, W[idx])
#     return cert_list[idx], idx, W[idx]


#################################################################################################################

######## Remember to change this to 1 when running ellipse ######################
previous_target_shape = 1 # initialize the shape flag as 1 (1 is circle and 2 is ellipse)
target_array = np.zeros(CBF_n)
target_array[previous_target_shape-1] = 1

# Default shape (begin with circle)         
Delta_cur = np.array([1.0,0.0]) # current Delta array
lamb = np.array([1.0,0.0]) # current lambda array (storing the current shape)
Delta = 0 # Delta

# Initialize the transition variables
transition_in_progress = False
# start_time = None
T = 1  # Duration for the morphing transition in seconds
exp_start_time = time.time()

# Create SI to UNI dynamics tranformation
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# define x initially
x = r.get_poses()
thetas = x[2,:]
L = 0.05
# g = r.axes.scatter(x[0,:]+L*np.cos(x[2,:]), x[1,:]+L*np.sin(x[2,:]), s=np.pi/4*safety_radius_marker_size, marker='o', facecolors='none',edgecolors=CM,linewidth=3)


# Create Goal Point Markers
goal_marker_size_m = 0.1
font_size = determine_font_size(r,0.1)
line_width = 5
marker_size_goal = determine_marker_size(r,goal_marker_size_m)

#Text with goal identification
goal_caption = ['G{0}'.format(ii) for ii in range(goal_points.shape[1])]

#Arrow for desired orientation (optional)
# goal_orientation_arrows = [r.axes.arrow(goal_points[0,ii], goal_points[1,ii], goal_marker_size_m*np.cos(goal_points[2,ii]), goal_marker_size_m*np.sin(goal_points[2,ii]), width = 0.01, length_includes_head=True, color = CM[ii,:], zorder=-2)
# for ii in range(goal_points.shape[1])]

#Plot text for caption
goal_points_text = [r.axes.text(goal_points[0,ii], goal_points[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-3)
for ii in range(goal_points.shape[1])]
goal_markers = [r.axes.scatter(goal_points[0,ii], goal_points[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-3)
for ii in range(goal_points.shape[1])]

r.step()
# r.step_no_error() (for robotarium with 20 agents)

# Initialize a list to keep track of the scatter objects
g_objects = []

# Initialize a list to keep track of the robots' trajectories
trajectories = {i: [] for i in range(N)}

# Initialize empty lists to store norm values
norm_dxi_cir_list = []
norm_dxi_ellip_list = []

## initialize for loop speed calculation
start_time = None # timer variable in loop
dt = None
prev_time = None

# While the goal is not reached
while(1):
    # for i in range(iterations):
        # Get poses of agents
        x = r.get_poses()

        # Store current positions in the trajectory list
        for i in range(N):
            trajectories[i].append(x[:, i].copy())  # Store the (x, y) position

        # Angles
        thetas = x[2,:]
        # To compare distances, only take the first two elements of our pose array.(look ahead dynamics)
        x_si = uni_to_si_states(x)

        # Use a position controller to drive to the goal position
        dxi = si_position_controller(x_si,goal_points[:2,:])

        ########################### barrier type ######################################
        # Use the barrier certificates to make sure that the agents don't collide
        # Generating safe inputs

        dxi_cir = si_barrier_cert_cir(dxi, x_si)                # the first barrier being circular
        # dxi_cir = si_barrier_cert_ellip(dxi, x_si,thetas)     # the first barrier being elliptical 
        dxi_ellip = si_barrier_cert_ellip(dxi, x_si,thetas)     # the second barrier being elliptical
        # dxi_ellip = si_barrier_cert_cir(dxi, x_si)            # the second barrier being circular
        
        
        ############################# selection ########################################
        # Use the second single-integrator-to-unicycle mapping to map to unicycle
        dxu_cir = si_to_uni_dyn(dxi_cir, x) # circular
        dxu_ellip = si_to_uni_dyn(dxi_ellip, x) # elliptical
        
        # Default shape
        # dxu = dxu_cir
        # For target shape, circle =1, ellipse=2
        norm_dxi_cir = np.linalg.norm(dxi_cir,ord=2)
        norm_dxi_ellip = np.linalg.norm(dxi_ellip,ord=2)
        ##TODO find the norm of the actual input in si

        # Append the norms to the lists for post-processing
        norm_dxi_cir_list.append(norm_dxi_cir)
        norm_dxi_ellip_list.append(norm_dxi_ellip)

        # Finding s_t, which is the shape we are morphing to
        target_shape = np.argmax([norm_dxi_cir,norm_dxi_ellip]) + 1 # s_t (shape to morph into) (1 is circle, 2 is ellipse)
        # print("index:",target_shape)

        ## Set target shape array
        target_array[:] = 0 # reset to 0
        # print("target_array_zeros",target_array)
        target_array[target_shape - 1] = 1 # target array, the shape we want to morph into
        # print("target_array_actual",target_array)
        # print("Delta_cur",Delta_cur)

        ######################################################################################
        if prev_time is None:
            prev_time = time.time() # start timer
        now = time.time() # start counter
        dt = now - prev_time 
        # eta = np.sqrt(2)/T*dt # rate of change
        # # print("dt",dt)
        # # print(np.linalg.norm((target_array-Delta_cur),ord=2))

        # # if the change gets to a terminal shape, or exceeds it
        # if np.linalg.norm(target_array-Delta_cur,ord=2) <= eta:
        #     Delta_target = target_array.copy() # complete transformation to a terminal shape
        # else:
        #     # we morph into the desired CBF 
        #     Delta_target = Delta_cur + eta* (target_array-Delta_cur) # used to calculate h3, discretized change
         
        # Delta_dot = 1/T # compute delta dot
        # print("Delta_target:",Delta_target)
        # print("Delta_cur[1]:",Delta_cur[1])

        # For plotting CBF shapes, the a and b currently 
        b_cur = (1-Delta) *(lamb[0] * 0.25 + lamb[1] * 0.20) + Delta * 0.20   # Interpolate ellipse width to circle radius
        a_cur = 0.25  # Keep a constant, or you can interpolate if needed
        # print("Delta sum:",Delta_cur[0]+Delta_cur[1])

        prev_time = time.time() # record time
        ##########################################################################################
        ## If target has changed, reset Delta and update lambda

        # print("previous_target_shape",previous_target_shape)
        
        if previous_target_shape != target_shape:
            for i in range(CBF_n):
                if i == target_shape - 1:
                    lamb[i] = (1-Delta)*lamb[i] + Delta
                else:
                    lamb[i] = (1-Delta)*lamb[i]
            Delta = 0
            previous_target_shape = target_shape
        else:
            Delta = np.clip(Delta + 1/T*dt, 0, 1)  # update Delta   

        # print("Delta",Delta)
        # print("lambda",lamb)

        ##########################################################################################
        # print("a_cur:",a_cur)
        # print("b_cur:",b_cur)

        ## Delta cur is used to get the current convex combination CBF, and Delta target is used to calculate h3 (time varying CBF)!

        # si_barrier_cert_tv, idx_sel, (w1_sel, w2_sel) = pick_cert_for_Delta(Delta_cur, target_shape)
        si_barrier_cert_tv = create_single_integrator_barrier_certificate_time_varying(Delta=Delta,lamb=lamb,target_shape=target_shape,T=T
                                                                                       ,barrier_gain=1,safety_radius=radius
                                                                                       ,safety_a=a,safety_b=b,a_cur=a_cur,b_cur=b_cur)  

        # si_barrier_cert_tv = create_single_integrator_barrier_certificate_ellipse(barrier_gain=1,safety_a=a,safety_b=b)
        dxi_tv = si_barrier_cert_tv(dxi, x_si, thetas)  
        dxu_tv = si_to_uni_dyn(dxi_tv, x)      
        dxu = dxu_tv
        # dxu = dxu_ellip
        ## Delta cur has ([circle,ellipse])
        # Delta_cur = Delta_target # update Delta current     
        # Remove previous scatter plot markers
        for g in g_objects:
            g.remove()

        # Clear the list after removing markers
        g_objects = []  

        ## Update Plotted Visualization
        # g.set_offsets(x[:2,:].T+np.array([L*np.cos(x[2,:]),L*np.sin(x[2,:])]).T)

        ## This updates the marker sizes if the figure window size is changed. 
        # g.set_sizes([determine_marker_size(r,safety_radius)])

        # g_objects.append(g)

        ############################# for plotting only #######################################
        # error = x[:] - x[:]
        # # circular CBF
        # h_circ = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)
        # # ellipitical CBF
        # error_1 = (error[0]*np.cos(theta[j])+error[1]*np.sin(theta[j])) / safety_a
        # error_2 = (error[0]*np.sin(theta[j])-error[1]*np.cos(theta[j])) / safety_b 
        # h_ellip = error_1**2 + error_2**2 - 1 
        # # calculate the current h (convex combination)
        # h_cur = lamb[0] * h_circ +  lamb[1] * h_ellip 
        # if target_shape == 1:
        #         # calculate h_3 
        #         h_tv = (1-Delta) * h_cur + Delta * h_circ 
        # if target_shape == 1:
        
        #########################################################################################
        # Create and add ellipses to the axes
        for i in range(N):
            ellipse = Ellipse(xy=(x[0, i] + L * np.cos(x[2, i]), x[1, i] + L * np.sin(x[2, i])),
                            width=a_cur*0.9, height=b_cur*0.9, angle=np.degrees(thetas[i]),
                            facecolor='none', edgecolor=CM[i], linewidth=2)
            r.axes.add_patch(ellipse)
            g_objects.append(ellipse)  # Keep track of the patches       

        # Set the velocities by mapping the single-integrator inputs to unciycle inputs
        r.set_velocities(np.arange(N), dxu)

        # Stopping cirterion (if goal is reached)
        if(np.linalg.norm(goal_points[:2,:] - x_si) < 0.08):
            break

        # Iterate the simulation
        r.step()
        # r.step_no_error() (for robotarium with 20 agents)

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()


# Convert lists to a single 2D NumPy array
u_norms_array = np.column_stack((norm_dxi_cir_list, norm_dxi_ellip_list))

#Save Data
print(time.time() - exp_start_time)
np.save('trajectories', trajectories)
np.save("u_norms", u_norms_array)

## plot block

## Plotting the position trajectories
# print("Preparing to plot trajectories...")
## Set the font globally to Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
## Enable LaTeX plotting
# plt.rc('text', usetex=True)

# plt.figure(figsize=(8, 8))
# for i in range(N):
#     trajectory = np.array(trajectories[i])
#     plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Robot {i + 1}', color=CM[i],linewidth=3)

# plt.scatter(goal_points[0, :], goal_points[1, :], color=CM, marker='*', s=200, label='Goals',linewidth=3)

## Add 10 cm (0.1 m) dashed circles around each goal point
# for i in range(goal_points.shape[1]):  # Loop over goal points
#     circle = plt.Circle((goal_points[0, i], goal_points[1, i]), 0.1, color=CM[i], fill=False, linestyle='dashed', linewidth=3)
#     plt.gca().add_patch(circle)  # Add the circle to the plot

## Increase font sizes for title, labels, and legend
# # plt.title('Robot Trajectories', fontsize=40)
# plt.xlabel('$$x (m)$$', fontsize=24)
# plt.ylabel('$$y (m)$$', fontsize=24)
# plt.xticks(fontsize=18)  # Font size for x-axis ticks
# plt.yticks(fontsize=18)  # Font size for y-axis ticks

## Adjust legend positioning to fit well within the plot
# # legend = plt.legend(fontsize=12, loc='upper left') 
# # legend.set_draggable(True)  # Make the legend draggable
# plt.savefig("plot.png", dpi=600)
# # plt.show(block=True)  # Keep the plot window open
# print("Plotting complete.")


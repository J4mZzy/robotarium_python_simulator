import rps.robotarium as robotarium
from rps.utilities.transformations import *
from barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import time

'''This code is for the multi-robot parallel swapping simulations/experiments'''
'''Not used for paper revision'''

# The robots will never reach their goal points so set iteration number (not used here)
iterations = 600

## The areana is bounded between x \in (-1.6,1.6)  y\in (-1,1) 

# Number of robots
N = 11 # 2,4,8,11,16,20

#############################################################################################
rect_width = 1.6  # Width of the rectangle
rect_height = 1.2  # Height of the rectangle

# Initialize positions
initial_x = np.zeros(N)
initial_y = np.zeros(N)
initial_heading = np.zeros(N)

# Cases for different number of robots 
if N == 2:
    # For the two-robot case
    initial_x[0] = -rect_width / 2  # Left side x-coordinate
    initial_y[0] = 0  # Centered vertically
    initial_heading[0] = 0  # Facing right

    initial_x[1] = rect_width / 2  # Right side x-coordinate
    initial_y[1] = 0  # Centered vertically
    initial_heading[1] = np.pi  # Facing left
elif N>=10 and np.mod(N,2)==0:
    # For two columns on each side
    spacing_x = rect_width / 4  # Horizontal spacing between columns
    spacing_y = rect_height / (N // 4 - 1)  # Vertical spacing between robots in each column

    # Left side (robots facing right)
    for i in range(N // 4):
        initial_x[i] = -rect_width / 2  # Left column x-coordinate
        initial_y[i] = rect_height / 2 - i * spacing_y  # Distribute vertically
        initial_heading[i] = 0  # Facing right

        j = i + N // 4
        initial_x[j] = -rect_width / 2 + spacing_x  # Right column x-coordinate (on the left side)
        initial_y[j] = rect_height / 2 - i * spacing_y  # Same vertical position
        initial_heading[j] = 0  # Facing right

    # Right side (robots facing left)
    for i in range(N // 4):
        k = i + N // 2
        initial_x[k] = rect_width / 2  # Left column x-coordinate (on the right side)
        initial_y[k] = rect_height / 2 - i * spacing_y  # Distribute vertically
        initial_heading[k] = np.pi  # Facing left

        l = k + N // 4
        initial_x[l] = rect_width / 2 - spacing_x  # Right column x-coordinate (on the right side)
        initial_y[l] = rect_height / 2 - i * spacing_y  # Same vertical position
        initial_heading[l] = np.pi  # Facing left
elif  N<=10 and np.mod(N,2)==0:
    # Single column logic for N = 4,8
    spacing = rect_height / (N // 2 - 1)  # Vertical spacing between robots on each side

    # Left side (robots facing right)
    for i in range(N // 2):
        initial_x[i] = -rect_width / 2  # Left side x-coordinate
        initial_y[i] = rect_height / 2 - i * spacing  # Distribute vertically
        initial_heading[i] = 0  # Facing right

    # Right side (robots facing left)
    for i in range(N // 2, N):
        initial_x[i] = rect_width / 2  # Right side x-coordinate
        initial_y[i] = rect_height / 2 - (i - N // 2) * spacing  # Distribute vertically
        initial_heading[i] = np.pi  # Facing left
elif np.mod(N, 2) != 0:
    # Define vertical spacing for each side (left and right columns)
    left_column_robots = N // 2 + 1  # Left column gets one more robot if N is odd
    right_column_robots = N // 2  # Right column

    rect_height = rect_height*1.4

    spacing_left = rect_height / left_column_robots 
    spacing_left = spacing_left + 0.05 # Vertical spacing for left column
    spacing_right = rect_height / right_column_robots if right_column_robots > 0 else spacing_left  # Avoid division by zero

    # Staggered left-right distribution
    for i in range(N):
        if i % 2 == 0:  # Even index: Left column
            initial_x[i] = -rect_width / 2  # Left side x-coordinate
            initial_heading[i] = 0  # Facing right
            # Place robots in left column with even spacing
            initial_y[i] = rect_height / 2 - ((i // 2) * spacing_left)
        else:  # Odd index: Right column
            initial_x[i] = rect_width / 2  # Right side x-coordinate
            initial_heading[i] = np.pi  # Facing left
            # Stagger placement for the right column by shifting down slightly
            initial_y[i] = rect_height / 2 - ((i // 2) * spacing_right) - (spacing_left / 2)


# Combine initial positions into the required format (x, y, theta)
initial_conditions = np.array([initial_x, initial_y, initial_heading])

# Instantiate Robotarium object
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True, initial_conditions=initial_conditions)

# Define goal points for swapping behavior
goal_points = np.array([initial_x, initial_y, initial_heading])  # Start by setting goal points to current positions
goal_points[0, :] = -initial_conditions[0,:] 

#################################################################
# # radius of the circle robots are forming
# circle_radius = 0.9

# # Calculate initial positions in a circular formation
# theta = np.linspace(0, 2 * np.pi, N, endpoint=False)  # Angles for each robot
# initial_x = circle_radius * np.cos(theta)  # X coordinates
# initial_y = circle_radius * np.sin(theta)  # Y coordinates

# # Headings (facing inward)
# initial_heading = theta + np.pi  # Heading towards the center (add pi to point inward)

# # Combine initial positions into the required format (x, y, theta)
# initial_conditions = np.array([initial_x, initial_y, initial_heading])
# # Instantiate Robotarium object
# r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True, initial_conditions=initial_conditions)
# # Define goal points: for a swapping behavior, we can simply offset the current positions
# goal_points = np.array([initial_x, initial_y, theta])  # Start by setting goal points to the current positions
# goal_points[0, :] = -initial_conditions[0,:] 
# goal_points[1, :] = -initial_conditions[1,:]
##################################################################


# Plotting Parameters

# #GT color scheme
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
safety_radius = 0.15
safety_radius_marker_size = determine_marker_size(r,safety_radius) # Will scale the plotted markers to be the diameter of provided argument (in meters)
font_height_meters = 0.2
font_height_points = determine_font_size(r,font_height_meters) # Will scale the plotted font height to that of the provided argument (in meters)

# Create single integrator position controller
si_position_controller = create_si_position_controller()

# Initialize parameters
radius = 0.25
a = 0.25
b = 0.20
w = 0.40

############################################ CBF Library #######################################################
# We're working in single-integrator dynamics, and we don't want the robots
# to collide.  Thus, we're going to use barrier certificates (in a centrialized way)
CBF_n = 4 # how many CBFs we are using 
si_barrier_cert_cir = create_single_integrator_barrier_certificate(barrier_gain=10,safety_radius=radius)
si_barrier_cert_ellip = create_single_integrator_barrier_certificate_ellipse(barrier_gain=1,safety_a=a,safety_b=b)
si_barrier_cert_sqaure = create_single_integrator_barrier_certificate_square(barrier_gain=1,safety_width=w,norm=3)
si_barrier_cert_tri = create_single_integrator_barrier_certificate_triangle(barrier_gain=1)

######## remember to change this to 2 when running ellipse ######################
######## Remember to change this to 1 when running ellipse ######################
previous_target_shape = 1 # initialize the shape flag as 1 (1 is circle and 2 is ellipse)
target_array = np.zeros(CBF_n)
target_array[previous_target_shape-1] = 1

# Default shape (begin with circle)         
lamb = np.array([1.0,0.0,0.0,0.0]) # current lambda array (storing the current shape)
Delta = 0 # Delta

lamb_list = []
Delta_list = []


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
font_size = determine_font_size(r,0.1*0.8)
line_width = 3

marker_size_goal = determine_marker_size(r,goal_marker_size_m)

#Text with goal identification
goal_caption = ['G{0}'.format(ii) for ii in range(goal_points.shape[1])]
#Arrow for desired orientation
# goal_orientation_arrows = [r.axes.arrow(goal_points[0,ii], goal_points[1,ii], goal_marker_size_m*np.cos(goal_points[2,ii]), goal_marker_size_m*np.sin(goal_points[2,ii]), width = 0.01, length_includes_head=True, color = CM[ii,:], zorder=-2)
# for ii in range(goal_points.shape[1])]
#Plot text for caption
goal_points_text = [r.axes.text(goal_points[0,ii], goal_points[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-3)
for ii in range(goal_points.shape[1])]
goal_markers = [r.axes.scatter(goal_points[0,ii], goal_points[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-3)
for ii in range(goal_points.shape[1])]
# rectangle_box = r.axes.scatter(0, 0, s=marker_size_goal*50, marker='s', facecolors='none',edgecolors=(255/255, 0/255, 0/255),linewidth=line_width,zorder=-3)

r.step()
# r.step_no_error() (for robotarium with 20 agents)

# Initialize a list to keep track of the scatter objects
g_objects = []

# Initialize a list to keep track of the robots' trajectories
trajectories = {i: [] for i in range(N)}

# Initialize empty lists to store norm values
norm_dxi_cir_list = []
norm_dxi_ellip_list = []
norm_dxi_tv_list = []

## initialize for loop speed calculation
start_time = None # timer variable in loop
dt = None
prev_time = None

t = 0

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

        # Use the barrier certificates to make sure that the agents don't collide

        ########################### barrier type ######################################
        # Use the barrier certificates to make sure that the agents don't collide
        dxi_cir = si_barrier_cert_cir(dxi, x_si)                # the first barrier being circular
        # dxi_cir = si_barrier_cert_ellip(dxi, x_si, thetas)     # the first barrier being elliptical 
        dxi_ellip = si_barrier_cert_ellip(dxi, x_si, thetas)     # the second barrier being elliptical
        # dxi_ellip = si_barrier_cert_cir(dxi, x_si)            # the second barrier being circular


        dxi_square = si_barrier_cert_sqaure(dxi, x_si, thetas)                # square
        dxi_tri = si_barrier_cert_tri(dxi, x_si, thetas)

        ###############################################################################

        ############################# selection ########################################
        # Use the second single-integrator-to-unicycle mapping to map to unicycle
        dxu_cir = si_to_uni_dyn(dxi_cir, x) # circular
        dxu_ellip = si_to_uni_dyn(dxi_ellip, x) # elliptical
        # dxu_sqaure = si_to_uni_dyn(dxi_square, x) # square
        # dxu_tri = si_to_uni_dyn(dxi_tri, x) # elliptical
        
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

        ############################################ TEST #################################################
        if norm_dxi_cir >= norm_dxi_ellip:
            target_shape = 1
        elif norm_dxi_cir < norm_dxi_ellip:
            target_shape = 4 #square
        ###########################################################################################################################

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
        # print("dt",dt)
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
        
        ## Delta = sin(t) for t \in [0,pi/2), and = 1 if t >= \pi/2
        if previous_target_shape != target_shape:
            for i in range(CBF_n):
                if i == target_shape - 1:
                    lamb[i] = (1-Delta)*lamb[i] + Delta
                else:
                    lamb[i] = (1-Delta)*lamb[i]
            Delta = 0
            previous_target_shape = target_shape
            t = 0 # reset time
        else:
            t = t + dt # update time
            if 0 <= t < 1:
                Delta = np.clip(Delta + np.pi/2*np.sin(np.pi*t)*dt, 0, 1)  # update Delta   
            else:
                Delta = 1
        # print(t)
        # print("Delta",Delta)
        # print("lambda",lamb)

        lamb_list.append(lamb.copy())
        Delta_list.append(Delta)

        ##########################################################################################
        # print("a_cur:",a_cur)
        # print("b_cur:",b_cur)

        ## Delta cur is used to get the current convex combination CBF, and Delta target is used to calculate h3 (time varying CBF)!

        # si_barrier_cert_tv, idx_sel, (w1_sel, w2_sel) = pick_cert_for_Delta(Delta_cur, target_shape)
        si_barrier_cert_tv = create_single_integrator_barrier_certificate_time_varying(Delta=Delta,lamb=lamb,target_shape=target_shape,t=t
                                                                                       ,barrier_gain=1,safety_radius=radius
                                                                                       ,safety_a=a,safety_b=b)  

        # si_barrier_cert_tv = create_single_integrator_barrier_certificate_ellipse(barrier_gain=1,safety_a=a,safety_b=b)
        dxi_tv = si_barrier_cert_tv(dxi, x_si, thetas)  
        dxu_tv = si_to_uni_dyn(dxi_tv, x)      
        
        # dxu = dxu_tv
        dxu = dxu_sqaure

        norm_dxi_tv = np.linalg.norm(dxi_tv,ord=2)
        # Append the norms to the lists for post-processing
        norm_dxi_tv_list.append(norm_dxi_tv)

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
        # for i in range(N):
        #     ellipse = Ellipse(xy=(x[0, i] + L * np.cos(x[2, i]), x[1, i] + L * np.sin(x[2, i])),
        #                     width=a_cur*0.9, height=b_cur*0.9, angle=np.degrees(thetas[i]),
        #                     facecolor='none', edgecolor=CM[i], linewidth=2)
        #     r.axes.add_patch(ellipse)
        #     g_objects.append(ellipse)  # Keep track of the patches       

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
u_norms_array = np.column_stack((norm_dxi_cir_list, norm_dxi_ellip_list,norm_dxi_tv_list))

#Save Data
print(time.time() - exp_start_time)
np.save('trajectories', trajectories)
np.save("u_norms", u_norms_array)
np.save("lamb_list", lamb_list)
np.save("Delta_list", Delta_list)

# plot block

## Plotting the position trajectories
# print("Preparing to plot trajectories...")
## Set the font globally to Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
## Enable LaTeX plotting
# plt.rc('text', usetex=True)

# plt.figure(figsize=(10, 8))
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
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

# The robots will never reach their goal points so set iteration number (not used here)
iterations = 600

## The areana is bounded between x \in (-1.6,1.6)  y\in (-1,1) 

# Number of robots
N = 20 # 2,4,8,11,16,20

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
radius = 0.20
a = 0.25
b = 0.20

# We're working in single-integrator dynamics, and we don't want the robots
# to collide.  Thus, we're going to use barrier certificates (in a centrialized way)
si_barrier_cert_cir = create_single_integrator_barrier_certificate_with_boundary(barrier_gain=10,safety_radius=radius)
si_barrier_cert_ellip = create_single_integrator_barrier_certificate_with_boundary_ellipse(barrier_gain=0.1,safety_a=a,safety_b=b)
######## remember to change this to 2 when running ellipse ######################
prev_CBF_shape = 1 # initialize the shape flag as 1 (1 is circle and 2 is ellipse)

# Initialize the transition variables
transition_in_progress = False
start_time = None
transition_duration = 1  # Duration for the morphing transition in seconds
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
        # dxi_cir = si_barrier_cert_ellip(dxi, x_si,thetas)     # the first barrier being elliptical 
        dxi_ellip = si_barrier_cert_ellip(dxi, x_si,thetas)     # the second barrier being elliptical
        # dxi_ellip = si_barrier_cert_cir(dxi, x_si)            # the second barrier being circular
        ###############################################################################

        # Use the second single-integrator-to-unicycle mapping to map to unicycle
        # dynamics
        dxu_cir = si_to_uni_dyn(dxi_cir, x)
        dxu_ellip = si_to_uni_dyn(dxi_ellip, x)

        norm_dxi_cir = np.linalg.norm(dxi_cir,ord=2)
        norm_dxi_ellip = np.linalg.norm(dxi_ellip,ord=2)

        # Append the norms to the lists
        norm_dxi_cir_list.append(norm_dxi_cir)
        norm_dxi_ellip_list.append(norm_dxi_ellip)

        if not transition_in_progress:
            # for smooth transitions
            if norm_dxi_cir > norm_dxi_ellip:
                max_norm = norm_dxi_cir # keep the circle
                current_CBF_shape = 1  # Circle
            elif norm_dxi_cir < norm_dxi_ellip:
                max_norm = max(norm_dxi_cir,norm_dxi_ellip)
                current_CBF_shape = 2  # Ellipse
            elif norm_dxi_cir == norm_dxi_ellip:
                current_CBF_shape = prev_CBF_shape   # keep the shape
        
        # Check if the shape has changed
        if current_CBF_shape != prev_CBF_shape and not transition_in_progress:
            # Shape has changed, start the transition
            transition_in_progress = True
            start_time = time.time()  # Reset the start time for transition
            prev_CBF_shape = current_CBF_shape  # Update the flag with the new shape
        
        # If transition is in progress, calculate alpha and morph between shapes
        if transition_in_progress:
            time_elapsed = time.time() - start_time  # Calculate elapsed time
            alpha = np.clip(time_elapsed / transition_duration, 0, 1)  # Compute alpha
            
            if current_CBF_shape == 1:
                # Morph from ellipse to circle
                dxu = dxu_cir
                b = (1 - alpha) * 0.14 + alpha * 0.16  # Interpolate ellipse width to circle radius
                a = 0.16  # Keep a constant, or you can interpolate if needed
            elif current_CBF_shape == 2:
                # Morph from circle to ellipse
                dxu = dxu_ellip
                b = (1 - alpha) * 0.16 + alpha * 0.14  # Interpolate circle radius to ellipse width
                a = 0.16  # Keep a constant, or adjust if you want the height to morph too

            # If the transition is complete, stop the morphing
            if alpha >= 1:
                transition_in_progress = False
        else:
            if current_CBF_shape == 1:
                dxu = dxu_cir
                a = 0.16
                b = 0.16

            elif current_CBF_shape == 2:
                dxu = dxu_ellip
                a = 0.16
                b = 0.14
        # Remove previous scatter plot markers
        for g in g_objects:
            g.remove()

        # Clear the list after removing markers
        g_objects = []  

        # # Update Plotted Visualization
        # g.set_offsets(x[:2,:].T+np.array([L*np.cos(x[2,:]),L*np.sin(x[2,:])]).T)

        # # This updates the marker sizes if the figure window size is changed. 
        # g.set_sizes([determine_marker_size(r,safety_radius)])

        # g_objects.append(g)

        # Create and add ellipses to the axes
        for i in range(N):
            ellipse = Ellipse(xy=(x[0, i] + L * np.cos(x[2, i]), x[1, i] + L * np.sin(x[2, i])),
                            width=a, height=b, angle=np.degrees(thetas[i]),
                            facecolor='none', edgecolor=CM[i], linewidth=2)
            r.axes.add_patch(ellipse)
            g_objects.append(ellipse)  # Keep track of the patches
        

        # Set the velocities by mapping the single-integrator inputs to unciycle inputs
        r.set_velocities(np.arange(N), dxu)

        # Stopping cirterion 
        if(np.linalg.norm(goal_points[:2,:] - x_si) < 0.1):
            break

        # Iterate the simulation
        r.step()
        # r.step_no_error()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()

# Convert lists to a single 2D NumPy array
u_norms_array = np.column_stack((norm_dxi_cir_list, norm_dxi_ellip_list))

#Save Data
print(time.time() - exp_start_time)
np.save('trajectories', trajectories)
np.save("u_norms", u_norms_array)

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
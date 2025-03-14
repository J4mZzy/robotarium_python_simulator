import rps.robotarium as robotarium
from rps.utilities.transformations import *
from barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import time


'''This code is for the decentrailized implementation of the CBF selection algorithm for simulations/experiments
CAVEAT: IT IS STILL UNDER PROGRESS (FUTURE WORK), IT MAY NOT WORK!'''

# # Instantiate Robotarium object
# N = 4

# r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True, initial_conditions=np.array([[0.4,0.4,-0.4,-0.4],[0.4,-0.4,-0.4,0.4],[-np.pi*3/4,np.pi*3/4,np.pi/4,-np.pi/4]]))
# for riangle: np.array([[-0.2*np.sqrt(3),0.2*np.sqrt(3),0],[-0.2,-0.2,0.4],[np.pi/6,np.pi/18*15,-np.pi/2]])

# The robots will never reach their goal points so set iteration number (not used here)
iterations = 600

## The areana is bounded between x \in (-1.6,1.6) y\in (-1,1) 

# Define goal points outside of the arena
# goal_points = np.array(np.array([[-0.4,-0.4,0.4,0.4],[-0.4,0.4,0.4,-0.4],[-np.pi*3/4,np.pi*3/4,np.pi/4,-np.pi/4]]))  # go straight for each

# This portion of the code generates points on a circle enscribed in a 6x6 square
# that's centered on the origin.  The robots switch positions on the circle.
# Define the radius of the circle for robot initial positions
N = 10
circle_radius = 0.8

# Calculate initial positions in a circular formation
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)  # Angles for each robot
initial_x = circle_radius * np.cos(theta)  # X coordinates
initial_y = circle_radius * np.sin(theta)  # Y coordinates

# Calculate headings (facing inward)
initial_heading = theta + np.pi  # Heading towards the center (add pi to point inward)

# Combine initial positions into the required format (x, y, theta)
initial_conditions = np.array([initial_x, initial_y, initial_heading])
# Instantiate Robotarium object
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True, initial_conditions=initial_conditions)
# Define goal points: for a swapping behavior, we can simply offset the current positions
goal_points = np.array([initial_x, initial_y, theta])  # Start by setting goal points to the current positions
# Swap the diagonally opposite robots
for i in range(N//2):
    opposite_idx = (i + N//2) % N  # Index of the opposite robot
    
    # Store positions in a temporary variable to avoid overwriting
    temp_x, temp_y, temp_theta = goal_points[:, i]
    
    goal_points[0, i] = goal_points[0, opposite_idx]  # Swap x
    goal_points[1, i] = goal_points[1, opposite_idx]  # Swap y
    goal_points[2, i] = goal_points[2, opposite_idx]  # Swap theta
    
    goal_points[0, opposite_idx] = temp_x  # Assign temp values to opposite robot
    goal_points[1, opposite_idx] = temp_y
    goal_points[2, opposite_idx] = temp_theta

# Plotting Parameters

##GT color scheme
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


################for Robotarium#######################
# Set CM to be all black for N agents
CM = np.array([[0, 0, 0]] * N)
 

# Default Barrier Parameters
safety_radius = 0.12

safety_radius_marker_size = determine_marker_size(r,safety_radius) # Will scale the plotted markers to be the diameter of provided argument (in meters)
font_height_meters = 0.2
font_height_points = determine_font_size(r,font_height_meters) # Will scale the plotted font height to that of the provided argument (in meters)



# # Create unicycle position controller
# unicycle_position_controller = create_clf_unicycle_position_controller()

# Create single integrator position controller
si_position_controller = create_si_position_controller()

# We're working in single-integrator dynamics, and we don't want the robots
# to collide.  Thus, we're going to use barrier certificates (in a centrialized way)

# Initialize parameters
radius = 0.2
a= 0.3
b = 0.2


###############################################################################
## TODO: add a decentrailzed implementation here
## Each agent will treat the other agents as barriers  
# Initialize a list of decentralized barrier certificates for each agent
# the index corresponds to the local agent
si_barrier_certs_cir = [create_single_integrator_barrier_certificate_decentralized(agent_index, barrier_gain=100, safety_radius=radius) for agent_index in range(N)]
si_barrier_certs_ellip = [create_single_integrator_barrier_certificate_ellipse_decentralized(agent_index, barrier_gain=100,safety_a=a,safety_b=b) for agent_index in range(N)]
# print(np.shape(si_barrier_certs_cir))
prev_CBF_shape = np.ones(N) # initialize the shape flag as 1s (1 is circle and 2 is ellipse)
current_CBF_shape = np.ones(N)  # To be updated for each agent
###############################################################################

# Create SI to UNI dynamics tranformation
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# define x initially
x = r.get_poses() #(3xN) shape, x,y, theta, e.g., agent 1 has x[:,0]
# print(x[:,0])
# print(np.shape(x))
thetas = x[2,:]   
# print(thetas)
L = 0.05
# g = r.axes.scatter(x[0,:]+L*np.cos(x[2,:]), x[1,:]+L*np.sin(x[2,:]), s=np.pi/4*safety_radius_marker_size, marker='o', facecolors='none',edgecolors=CM,linewidth=3)

# Initialize the transition variables
transition_in_progress = [False]*N # N flags
start_time = np.zeros(N)
transition_duration = 0.5  # Duration for the morphing transition in seconds



# Create Goal Point Markers
goal_marker_size_m = 0.1
font_size = determine_font_size(r,0.05)
line_width = 3

marker_size_goal = determine_marker_size(r,goal_marker_size_m)

#Text with goal identification
goal_caption = ['G{0}'.format(ii) for ii in range(goal_points.shape[1])]
#Arrow for desired orientation
goal_orientation_arrows = [r.axes.arrow(goal_points[0,ii], goal_points[1,ii], goal_marker_size_m*np.cos(goal_points[2,ii]), goal_marker_size_m*np.sin(goal_points[2,ii]), width = 0.01, length_includes_head=True, color = CM[ii,:], zorder=-2)
for ii in range(goal_points.shape[1])]
#Plot text for caption
goal_points_text = [r.axes.text(goal_points[0,ii], goal_points[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-3)
for ii in range(goal_points.shape[1])]
goal_markers = [r.axes.scatter(goal_points[0,ii], goal_points[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-3)
for ii in range(goal_points.shape[1])]


r.step()


# Initialize a list to keep track of the scatter objects
g_objects = []

# Initialize a list to keep track of the robots' trajectories
trajectories = {i: [] for i in range(N)}

# While the goal is not reached
while(1):
    # for i in range(iterations):
        # Get poses of agents
        x = r.get_poses()

        # Angles
        thetas = x[2,:]
        # thetas=np.zeros_like(thetas)  # all-zeros lol

        # Remove previous scatter plot markers
        for g in g_objects:
            g.remove()

        # Clear the list after removing markers
        g_objects = []  

        # Store current positions in the trajectory list
        for i in range(N):
            trajectories[i].append(x[:2, i].copy())  # Store the (x, y) position

        # To compare distances, only take the first two elements of our pose array.(look ahead dynamics)
        x_si = uni_to_si_states(x)

        # Use a position controller to drive to the goal position
        dxi = si_position_controller(x_si,goal_points[:2,:])
        # print(dxi)

        # Initialize an array to store the modified control inputs after applying barrier certificates
        dxi_cir = np.zeros((2, N))
        dxi_ellip = np.zeros((2, N))
        dxi_final = np.zeros((2, N))  # This will store the result with the greater norm
        # print(np.shape(dxi_cir))
        dxu = np.zeros((2, N))  # Store final unicycle control inputs
        # Apply decentralized barrier certificates for each agent
        for agent_index in range(N):
            # print(np.shape(si_barrier_certs_cir[agent_index](dxi, x_si)))
            dxi_cir[:,agent_index] = si_barrier_certs_cir[agent_index](dxi, x_si).reshape(2)
            # dxi_ellip[:,agent_index] = si_barrier_certs_ellip[agent_index](dxi, x_si,thetas).reshape(2)
            try:
                # Attempt to apply the barrier certificate for the ellipse shape
                dxi_ellip[:, agent_index] = si_barrier_certs_ellip[agent_index](dxi, x_si, thetas).reshape(2)
            except ValueError as e:
                # If an error occurs, set the control input for this agent to zeros
                print(f"Error for agent {agent_index}: {e}")
                dxi_ellip[:, agent_index] = np.zeros(2)  # Set the control input to zero for this agent

            # Convert dxi to unicycle dynamics
            dxu_cir = si_to_uni_dyn(dxi_cir, x)
            dxu_ellip = si_to_uni_dyn(dxi_ellip, x)

            # Calculate the norms of each control input
            norm_dxi_cir = np.linalg.norm(dxi_cir[:, agent_index],ord=2)
            norm_dxi_ellip = np.linalg.norm(dxi_ellip[:, agent_index],ord=2)

            # If no transition is in progress for this agent, determine the current shape
            if not transition_in_progress[agent_index]:
                if norm_dxi_cir >= norm_dxi_ellip:
                    current_CBF_shape[agent_index] = 1  # Circle
                else:
                    current_CBF_shape[agent_index] = 2  # Ellipse

            # Check if the shape has changed for this agent
            if current_CBF_shape[agent_index] != prev_CBF_shape[agent_index] and not transition_in_progress[agent_index]:
                # Shape has changed, start the transition for this agent
                transition_in_progress[agent_index] = True
                start_time[agent_index] = time.time()  # Reset the start time for transition
                prev_CBF_shape[agent_index] = current_CBF_shape[agent_index]  # Update the flag with the new shape

            # If transition is in progress for this agent, calculate alpha and morph between shapes
            if transition_in_progress[agent_index]:
                time_elapsed = time.time() - start_time[agent_index]  # Calculate elapsed time for this agent
                alpha = np.clip(time_elapsed / transition_duration, 0, 1)  # Compute alpha (progress in transition)

                # Morph between shapes based on current shape
                if current_CBF_shape[agent_index] == 1:
                    # Morph from ellipse to circle
                    dxu[:, agent_index] = (1 - alpha) * dxu_ellip[:, agent_index] + alpha * dxu_cir[:, agent_index]
                    a = (1 - alpha) * 0.20 + alpha * 0.15  # Interpolate ellipse width to circle radius
                    b = 0.15  # Keep height constant, or adjust if needed
                elif current_CBF_shape[agent_index] == 2:
                    # Morph from circle to ellipse
                    dxu[:, agent_index] = (1 - alpha) * dxu_cir[:, agent_index] + alpha * dxu_ellip[:, agent_index]
                    a = (1 - alpha) * 0.15 + alpha * 0.20  # Interpolate circle radius to ellipse width
                    b = 0.15  # Keep height constant, or adjust if needed

                # If the transition is complete for this agent, stop the morphing
                if alpha >= 1:
                    transition_in_progress[agent_index] = False

            # If no transition is happening, simply apply the current shape control input
            else:
                if current_CBF_shape[agent_index] == 1:
                    dxu[:, agent_index] = dxu_cir[:, agent_index]
                    a = 0.15
                    b = 0.15
                elif current_CBF_shape[agent_index] == 2:
                    dxu[:, agent_index] = dxu_ellip[:, agent_index]
                    a = 0.20
                    b = 0.15

            # Now draw the ellipse or circle for this agent based on its current shape
            ellipse = Ellipse(
                xy=(x[0, agent_index] + L * np.cos(x[2, agent_index]), x[1, agent_index] + L * np.sin(x[2, agent_index])),
                width=a, height=b, angle=np.degrees(thetas[agent_index]),
                facecolor='none', edgecolor=CM[agent_index], linewidth=2
            )
            r.axes.add_patch(ellipse)
            g_objects.append(ellipse)  # Keep track of the patches for updating later        

        # Set the velocities by mapping the single-integrator inputs to unciycle inputs
        r.set_velocities(np.arange(N), dxu)

        # Stopping cirterion 
        if(np.linalg.norm(goal_points[:2,:] - x_si) < 0.02):
            break

        # Iterate the simulation
        r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()


## plot block

# # Plotting the position trajectories
# print("Preparing to plot trajectories...")
# plt.figure(figsize=(10, 10))
# for i in range(N):
#     trajectory = np.array(trajectories[i])
#     plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Robot {i + 1}', color=CM[i],linewidth=3)

# plt.scatter(goal_points[0, :], goal_points[1, :], color=CM, marker='*', s=200, label='Goals',linewidth=3)
# # Increase font sizes for title, labels, and legend
# plt.title('Robot Trajectories', fontsize=20)
# plt.xlabel('X Position', fontsize=18)
# plt.ylabel('Y Position', fontsize=18)
# plt.xticks(fontsize=16)  # Font size for x-axis ticks
# plt.yticks(fontsize=16)  # Font size for y-axis ticks

# # Adjust legend positioning to fit well within the plot
# legend = plt.legend(fontsize=12, loc='upper left') 
# legend.set_draggable(True)  # Make the legend draggable

# plt.show(block=True)  # Keep the plot window open
# print("Plotting complete.")
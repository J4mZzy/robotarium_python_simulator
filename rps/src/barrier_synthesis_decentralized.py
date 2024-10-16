import rps.robotarium as robotarium
from rps.utilities.transformations import *
from barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import time

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
N = 16
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


CM = np.random.rand(N,3) # Random Colors
 

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
radius = 0.22
a= 0.3
b = 0.22

# Initialize a list of decentralized barrier certificates for each agent
si_barrier_certs_cir = [create_single_integrator_barrier_certificate_decentralized(agent_index, barrier_gain=1000, safety_radius=radius) for agent_index in range(N)]
# si_barrier_cert_ellip = create_single_integrator_barrier_certificate_ellipse(barrier_gain=100,safety_a=a,safety_b=b)

###############################################################################
## TODO: add a decentrailzed implementation here
## Each agent will treat the other agents as barriers  

###############################################################################

# Create SI to UNI dynamics tranformation
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# define x initially
x = r.get_poses()
thetas = x[2,:]
L = 0.05
# g = r.axes.scatter(x[0,:]+L*np.cos(x[2,:]), x[1,:]+L*np.sin(x[2,:]), s=np.pi/4*safety_radius_marker_size, marker='o', facecolors='none',edgecolors=CM,linewidth=3)


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

        # Initialize a velocities variable
        si_velocities = np.zeros((2, N))
        # Get poses of agents
        x = r.get_poses()

        # Store current positions in the trajectory list
        for i in range(N):
            trajectories[i].append(x[:2, i].copy())  # Store the (x, y) position

        # Angles
        thetas = x[2,:]
        # thetas=np.zeros_like(thetas)  # all-zeros lol

        # To compare distances, only take the first two elements of our pose array.(look ahead dynamics)
        x_si = uni_to_si_states(x)

        # Use a position controller to drive to the goal position
        dxi = si_position_controller(x_si,goal_points[:2,:])

        # Initialize an array to store the modified control inputs after applying barrier certificates
        dxi_cir = np.zeros((2, N))  # Assuming it's for 2D control inputs (x, y)

        # Apply decentralized barrier certificates for each agent
        for agent_index in range(N):
            dxi_cir[agent_index] = si_barrier_certs_cir[agent_index](dxi, x_si)


        ###############################################################################
        ## TODO: add a decentrailzed implementation here

        ###############################################################################

        # Use the second single-integrator-to-unicycle mapping to map to unicycle
        # dynamics
        # dxu_cir = si_to_uni_dyn(dxi_cir, x)
        # dxu_ellip = si_to_uni_dyn(dxi_ellip, x)

        # norm_dxu_cir = np.linalg.norm(dxu_cir,ord=2)
        # norm_dxu_ellip = np.linalg.norm(dxu_ellip,ord=2)

        # # for smooth transitions
        # if norm_dxu_cir >= norm_dxu_ellip:
        #     max_norm = norm_dxu_cir # keep the circle
        # else:
        #     max_norm = max(norm_dxu_cir,norm_dxu_ellip)

        # # Remove previous scatter plot markers
        # for g in g_objects:
        #     g.remove()

        # # Clear the list after removing markers
        # g_objects = []  

        # if max_norm == norm_dxu_cir:
        #     dxu = dxu_cir
        #     a = 0.17
        #     b = 0.17
        #     # g = r.axes.scatter(x[0,:]+L*np.cos(x[2,:]), x[1,:]+L*np.sin(x[2,:]), s=np.pi/4*safety_radius_marker_size, marker='o', facecolors='none',edgecolors=CM,linewidth=3)

        # elif max_norm == norm_dxu_ellip:
        #     dxu = dxu_ellip
        #     a = 0.22
        #     b = 0.17
        #     # g = r.axes.scatter(x[0,:]+L*np.cos(x[2,:]), x[1,:]+L*np.sin(x[2,:]), s=np.pi/4*safety_radius_marker_size, marker='D', facecolors='none',edgecolors=CM,linewidth=3)

        # # Update Plotted Visualization
        # g.set_offsets(x[:2,:].T+np.array([L*np.cos(x[2,:]),L*np.sin(x[2,:])]).T)

        # # This updates the marker sizes if the figure window size is changed. 
        # g.set_sizes([determine_marker_size(r,safety_radius)])

        # g_objects.append(g)

        # Map the modified single-integrator inputs to unicycle dynamics
        dxu = si_to_uni_dyn(dxi_cir, x)

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
        if(np.linalg.norm(goal_points[:2,:] - x_si) < 0.02):
            break

        # Iterate the simulation
        r.step()

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()


## plot block

# Plotting the position trajectories
print("Preparing to plot trajectories...")
plt.figure(figsize=(10, 10))
for i in range(N):
    trajectory = np.array(trajectories[i])
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Robot {i + 1}', color=CM[i],linewidth=3)

plt.scatter(goal_points[0, :], goal_points[1, :], color=CM, marker='*', s=200, label='Goals',linewidth=3)
# Increase font sizes for title, labels, and legend
plt.title('Robot Trajectories', fontsize=20)
plt.xlabel('X Position', fontsize=18)
plt.ylabel('Y Position', fontsize=18)
plt.xticks(fontsize=16)  # Font size for x-axis ticks
plt.yticks(fontsize=16)  # Font size for y-axis ticks

# Adjust legend positioning to fit well within the plot
legend = plt.legend(fontsize=12, loc='upper left') 
legend.set_draggable(True)  # Make the legend draggable

plt.show(block=True)  # Keep the plot window open
print("Plotting complete.")
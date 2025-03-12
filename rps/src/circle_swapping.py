import rps.robotarium as robotarium
from rps.utilities.transformations import *
from barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import time


# The robots will never reach their goal points so set iteration number (not used here)
iterations = 600

## The areana is bounded between x \in (-1.6,1.6)  y\in (-1,1) 

# Define goal points outside of the arena
# goal_points = np.array(np.array([[-0.4,-0.4,0.4,0.4],[-0.4,0.4,0.4,-0.4],[-np.pi*3/4,np.pi*3/4,np.pi/4,-np.pi/4]]))  # go straight for each

# This portion of the code generates points on a circle enscribed in a 6x6 square
# that's centered on the origin.  The robots switch positions on the circle.
# Define the radius of the circle for robot initial positions
N = 20 # 2,4,8,11,16,20
circle_radius = 0.9

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
# for i in range(N//2):
#     opposite_idx = (i + N//2) % N  # Index of the opposite robot
    
#     # Store positions in a temporary variable to avoid overwriting
#     temp_x, temp_y, temp_theta = goal_points[:, i]
    
#     goal_points[0, i] = goal_points[0, opposite_idx]  # Swap x
#     goal_points[1, i] = goal_points[1, opposite_idx]  # Swap y
#     goal_points[2, i] = goal_points[2, opposite_idx]  # Swap theta
    
#     goal_points[0, opposite_idx] = temp_x  # Assign temp values to opposite robot
#     goal_points[1, opposite_idx] = temp_y
#     goal_points[2, opposite_idx] = temp_theta

goal_points[0, :] = -initial_conditions[0,:] 
goal_points[1, :] = -initial_conditions[1,:]

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


################for Robotarium#######################
# Set CM to be all black for N agents
# CM = np.array([[0, 0, 0]] * N)
 

# Default Barrier Parameters
safety_radius = 0.15

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
radius = 0.25
a = 0.25
b = 0.20

si_barrier_cert_cir = create_single_integrator_barrier_certificate(barrier_gain=10,safety_radius=radius)
si_barrier_cert_ellip = create_single_integrator_barrier_certificate_ellipse(barrier_gain=0.1,safety_a=a,safety_b=b)
######## remember to change this to 2 when running ellipse ######################
prev_CBF_shape = 2 # initialize the shape flag as 1 (1 is circle and 2 is ellipse)

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
font_size = determine_font_size(r,0.1)
line_width = 5

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

r.step()
# r.step_no_error()


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
        # thetas=np.zeros_like(thetas)  # all-zeros lol

        # To compare distances, only take the first two elements of our pose array.(look ahead dynamics)
        x_si = uni_to_si_states(x)

        # Use a position controller to drive to the goal position
        dxi = si_position_controller(x_si,goal_points[:2,:])

        ########################### barrier type ######################################
        # Use the barrier certificates to make sure that the agents don't collide
        # dxi_cir = si_barrier_cert_cir(dxi, x_si)
        dxi_cir = si_barrier_cert_ellip(dxi, x_si,thetas)
        dxi_ellip = si_barrier_cert_ellip(dxi, x_si,thetas)
        # dxi_ellip = si_barrier_cert_cir(dxi, x_si) 

        ############### progress? ######################

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
        if(np.linalg.norm(goal_points[:2,:] - x_si) < 0.08):
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

# # Plotting the position trajectories
# print("Preparing to plot trajectories...")
# # Set the font globally to Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
# # Enable LaTeX plotting
# plt.rc('text', usetex=True)

# plt.figure(figsize=(8, 8))
# for i in range(N):
#     trajectory = np.array(trajectories[i])
#     plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Robot {i + 1}', color=CM[i],linewidth=3)

# plt.scatter(goal_points[0, :], goal_points[1, :], color=CM, marker='*', s=200, label='Goals',linewidth=3)

# # Add 10 cm (0.1 m) dashed circles around each goal point
# for i in range(goal_points.shape[1]):  # Loop over goal points
#     circle = plt.Circle((goal_points[0, i], goal_points[1, i]), 0.1, color=CM[i], fill=False, linestyle='dashed', linewidth=3)
#     plt.gca().add_patch(circle)  # Add the circle to the plot

# # Increase font sizes for title, labels, and legend
# # plt.title('Robot Trajectories', fontsize=40)
# plt.xlabel('$$x (m)$$', fontsize=24)
# plt.ylabel('$$y (m)$$', fontsize=24)
# plt.xticks(fontsize=18)  # Font size for x-axis ticks
# plt.yticks(fontsize=18)  # Font size for y-axis ticks

# # Adjust legend positioning to fit well within the plot
# # legend = plt.legend(fontsize=12, loc='upper left') 
# # legend.set_draggable(True)  # Make the legend draggable
# plt.savefig("plot.png", dpi=600)
# # plt.show(block=True)  # Keep the plot window open
# print("Plotting complete.")


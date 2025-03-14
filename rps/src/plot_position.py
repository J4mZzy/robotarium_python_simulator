import rps.robotarium as robotarium
from rps.utilities.transformations import *
from barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import matplotlib.pyplot as plt


'''This code generates the plot for the positions of the robots during the experiment 
from reading the data saved through the simulations/experiments'''

# Number of robots
N = 20 # 2,4,8,11,16,20

#################################### Circle swapping ###############################################################
# Uncomment if the simulation/experiment is a circle swapping scenario, comment out the block if it's the other case

# circle_radius = 0.9

# # Calculate initial positions in a circular formation
# theta = np.linspace(0, 2 * np.pi, N, endpoint=False)  # Angles for each robot
# initial_x = circle_radius * np.cos(theta)  # X coordinates
# initial_y = circle_radius * np.sin(theta)  # Y coordinates

# # Calculate headings (facing inward)
# initial_heading = theta + np.pi  # Heading towards the center (add pi to point inward)

# # Combine initial positions into the required format (x, y, theta)
# initial_conditions = np.array([initial_x, initial_y, initial_heading])
# # Instantiate Robotarium object
# r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True, initial_conditions=initial_conditions)
# # Define goal points: for a swapping behavior, we can simply offset the current positions
# goal_points = np.array([initial_x, initial_y, theta])  # Start by setting goal points to the current positions

# goal_points[0, :] = -initial_conditions[0,:] 
# goal_points[1, :] = -initial_conditions[1,:]

#################################### Parallel swapping #############################################################
# Uncomment if the simulation/experiment is a parallel swapping scenario, comment out the block if it's the other case

rect_width = 1.6  # Width of the rectangle
rect_height = 1.2  # Height of the rectangle

# Calculate initial positions
initial_x = np.zeros(N)
initial_y = np.zeros(N)
initial_heading = np.zeros(N)
# spacing = rect_height / (N // 2 - 1)  # Vertical spacing between robots on each side

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

    spacing_left = rect_height / left_column_robots*1.2  # Vertical spacing for left column
    spacing_right = rect_height / right_column_robots*1.2   # Vertical spacing for right column
    # Staggered left-right distribution
    for i in range(N):
        if i % 2 == 0:  # Even index: Left column
            initial_x[i] = -rect_width / 2  # Left side x-coordinate
            initial_heading[i] = 0  # Facing right
            # Place robots in left column with even spacing
            initial_y[i] = rect_height / 2 - (i // 2) * spacing_left
        else:  # Odd index: Right column
            initial_x[i] = rect_width / 2  # Right side x-coordinate
            initial_heading[i] = np.pi  # Facing left
            # Place robots in right column with even spacing
            initial_y[i] = rect_height / 2 - ((i - 1) // 2) * spacing_right
######################################################################################################################

###################################### Initialization ################################################################

# Combine initial positions into the required format (x, y, theta)
initial_conditions = np.array([initial_x, initial_y, initial_heading])

# Instantiate Robotarium object
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True, initial_conditions=initial_conditions)

# Define goal points for swapping behavior
goal_points = np.array([initial_x, initial_y, initial_heading])  # Start by setting goal points to current positions
goal_points[0, :] = -initial_conditions[0,:] 

######################################################################################################################

cmap = plt.get_cmap("tab20")  # You can try "Set3", "tab10", or any other colormap

# Generate colors from the colormap for N agents
CM = cmap(np.linspace(0, 1, N))  # Generate N colors evenly spaced within the colormap


# Create Goal Point Markers
goal_marker_size_m = 0.1
font_size = determine_font_size(r,0.1)
line_width = 5
marker_size_goal = determine_marker_size(r,goal_marker_size_m)

# Load data
trajectories = np.load('trajectories.npy', allow_pickle=True).item()

############################################### Plot block ########################################################

# Plotting the position trajectories
print("Preparing to plot trajectories...")
# Set the font globally to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# Enable LaTeX plotting
plt.rc('text', usetex=True)
plt.figure(figsize=(9.5, 9))
for i in range(N):
    trajectory = np.array(trajectories[i]) # i-th agent 
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Robot {i + 1}', color=CM[i],linewidth=3)
plt.scatter(goal_points[0, :], goal_points[1, :], color=CM, marker='*', s=200, label='Goals',linewidth=3)

# Add 10 cm (0.1 m) dashed circles around each goal point
for i in range(goal_points.shape[1]):  # Loop over goal points
    circle = plt.Circle((goal_points[0, i], goal_points[1, i]), 0.1, color=CM[i], fill=False, linestyle='dashed', linewidth=3)
    plt.gca().add_patch(circle)  # Add the circle to the plot

# Increase font sizes for title, labels, and legend
plt.xlabel('$$x (m)$$', fontsize=28)
plt.ylabel('$$y (m)$$', fontsize=28)
plt.xticks(fontsize=18)  # Font size for x-axis ticks
plt.yticks(fontsize=18)  # Font size for y-axis ticks

# Adjust legend positioning to fit well within the plot
# legend = plt.legend(fontsize=12, loc='upper left') 
# legend.set_draggable(True)  # Make the legend draggable
# plt.savefig("pos_plot.png", dpi=600)

plt.show(block=True)  # Keep the plot window open
print("Plotting complete.")

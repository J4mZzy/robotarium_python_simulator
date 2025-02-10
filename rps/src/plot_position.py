import rps.robotarium as robotarium
from rps.utilities.transformations import *
from barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import matplotlib.pyplot as plt

# This portion of the code generates points on a circle enscribed in a 6x6 square
# that's centered on the origin.  The robots switch positions on the circle.
# Define the radius of the circle for robot initial positions
N = 4 # 2,4,8,11,16,20
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

goal_points[0, :] = -initial_conditions[0,:] 
goal_points[1, :] = -initial_conditions[1,:]


cmap = plt.get_cmap("tab20")  # You can try "Set3", "tab10", or any other colormap

# Generate colors from the colormap for N agents
CM = cmap(np.linspace(0, 1, N))  # Generate N colors evenly spaced within the colormap


# Create Goal Point Markers
goal_marker_size_m = 0.1
font_size = determine_font_size(r,0.1)
line_width = 5

marker_size_goal = determine_marker_size(r,goal_marker_size_m)


# load data
trajectories = np.load('trajectories.npy', allow_pickle=True).item()

# plot block

# Plotting the position trajectories
print("Preparing to plot trajectories...")
# Set the font globally to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# Enable LaTeX plotting
plt.rc('text', usetex=True)

plt.figure(figsize=(8, 8))
for i in range(N):
    trajectory = np.array(trajectories[i])
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=f'Robot {i + 1}', color=CM[i],linewidth=3)

plt.scatter(goal_points[0, :], goal_points[1, :], color=CM, marker='*', s=200, label='Goals',linewidth=3)

# Add 10 cm (0.1 m) dashed circles around each goal point
for i in range(goal_points.shape[1]):  # Loop over goal points
    circle = plt.Circle((goal_points[0, i], goal_points[1, i]), 0.1, color=CM[i], fill=False, linestyle='dashed', linewidth=3)
    plt.gca().add_patch(circle)  # Add the circle to the plot

# Increase font sizes for title, labels, and legend
# plt.title('Robot Trajectories', fontsize=40)
plt.xlabel('$$x (m)$$', fontsize=24)
plt.ylabel('$$y (m)$$', fontsize=24)
plt.xticks(fontsize=18)  # Font size for x-axis ticks
plt.yticks(fontsize=18)  # Font size for y-axis ticks

# Adjust legend positioning to fit well within the plot
# legend = plt.legend(fontsize=12, loc='upper left') 
# legend.set_draggable(True)  # Make the legend draggable
plt.savefig("pos_plot.png", dpi=600)
# plt.show(block=True)  # Keep the plot window open
print("Plotting complete.")

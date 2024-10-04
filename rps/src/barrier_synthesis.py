import rps.robotarium as robotarium
from rps.utilities.transformations import *
from barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import time

# Instantiate Robotarium object
N = 4

r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True, initial_conditions=np.array([[0.4,0.4,-0.4,-0.4],[0.4,-0.4,-0.4,0.4],[-np.pi*3/4,np.pi*3/4,np.pi/4,-np.pi/4]]))
# for riangle: np.array([[-0.2*np.sqrt(3),0.2*np.sqrt(3),0],[-0.2,-0.2,0.4],[np.pi/6,np.pi/18*15,-np.pi/2]])

# The robots will never reach their goal points so set iteration number
iterations = 800

## The areana is bounded between x \in (-1.6,1.6) y\in (-1,1) 

# Default Barrier Parameters
safety_radius = 0.17

# Plotting Parameters

##GT color scheme
Gold = np.array([179,163,105])/255
Navy = np.array([0,48,87])/255
piMile= np.array([214,219,212])/255
Black = np.array([0,0,0])
CM = np.vstack([Gold,Navy,piMile,Black])


# CM = np.random.rand(N,3) # Random Colors


safety_radius_marker_size = determine_marker_size(r,safety_radius) # Will scale the plotted markers to be the diameter of provided argument (in meters)
font_height_meters = 0.2
font_height_points = determine_font_size(r,font_height_meters) # Will scale the plotted font height to that of the provided argument (in meters)

# Define goal points outside of the arena
goal_points = np.array(np.array([[-0.4,-0.4,0.4,0.4],[-0.4,0.4,0.4,-0.4],[-np.pi*3/4,np.pi*3/4,np.pi/4,-np.pi/4]]))  # go straight for each 

# Create unicycle position controller
unicycle_position_controller = create_clf_unicycle_position_controller()

# Create barrier certificates to avoid collision
uni_barrier_cert = create_unicycle_barrier_certificate_ellipse(barrier_gain=10,safety_a=0.3,safety_b=0.2)
uni_barrier_cert2 = create_unicycle_barrier_certificate(barrier_gain=0.1,safety_radius=0.17)
# uni_barrier_cert3 = create_unicycle_barrier_certificate_diamond(barrier_gain=10,safety_radius=0.12)



# define x initially
x = r.get_poses()
g = r.axes.scatter(x[0,:], x[1,:], s=np.pi/4*safety_radius_marker_size, marker='o', facecolors='none',edgecolors=CM,linewidth=7)

# Create Goal Point Markers

goal_marker_size_m = 0.08
font_size = determine_font_size(r,0.05)
line_width = 3

marker_size_goal = determine_marker_size(r,goal_marker_size_m)

#Text with goal identification
goal_caption = ['G{0}'.format(ii) for ii in range(goal_points.shape[1])]
#Arrow for desired orientation
goal_orientation_arrows = [r.axes.arrow(goal_points[0,ii], goal_points[1,ii], goal_marker_size_m*np.cos(goal_points[2,ii]), goal_marker_size_m*np.sin(goal_points[2,ii]), width = 0.02, length_includes_head=True, color = CM[ii,:], zorder=-2)
for ii in range(goal_points.shape[1])]
#Plot text for caption
goal_points_text = [r.axes.text(goal_points[0,ii], goal_points[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-3)
for ii in range(goal_points.shape[1])]
goal_markers = [r.axes.scatter(goal_points[0,ii], goal_points[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-3)
for ii in range(goal_points.shape[1])]


r.step()

# While the number of robots at the required poses is less
# than N...
for i in range(iterations):

    # Get poses of agents
    x = r.get_poses()
    
    # Update Plotted Visualization
    g.set_offsets(x[:2,:].T)
    # This updates the marker sizes if the figure window size is changed. 
    # This should be removed when submitting to the Robotarium.
    g.set_sizes([determine_marker_size(r,safety_radius)])

    # Create single-integrator control inputs
    dxu = unicycle_position_controller(x, goal_points[:2][:])

    # Create safe control inputs (i.e., no collisions)
    dxu_1 = uni_barrier_cert(dxu, x)
    dxu_2 = uni_barrier_cert2(dxu, x)
    # dxu_3 = uni_barrier_cert3(dxu, x)

    norm_dxu_1 = np.linalg.norm(dxu_1,ord=2)
    norm_dxu_2 = np.linalg.norm(dxu_2,ord=2)
    # norm_dxu_3 = np.linalg.norm(dxu_3,ord=2)

    # Find the vector with the largest 2-norm
    max_norm = max(norm_dxu_1, norm_dxu_2)

    if max_norm == norm_dxu_1:
        dxu = dxu_1
    elif max_norm == norm_dxu_2:
        dxu = dxu_2
    # else:
    #     dxu = dxu_3


    # Set the velocities by mapping the single-integrator inputs to unciycle inputs
    r.set_velocities(np.arange(N), dxu)

    # Iterate the simulation
    r.step()


#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
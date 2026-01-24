#
# Circle Swap Experiment with obstacles using a singular barrier certificate
#

import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.misc import determine_marker_size, determine_font_size
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.transformations import create_si_to_uni_mapping

import numpy as np
import matplotlib.pyplot as plt
import time

from barrier_certificates import (
    SICircularBarrierCertificate,
    SIEllipticalBarrierCertificate,
    SISquareBarrierCertificate,
    SITriangleBarrierCertificate,
    SIDeltaBarrierCertificate
)
from graphing_utilities import init_hvis, update_hvis

import matplotlib as mpl
mpl.rcParams["path.simplify"] = False
mpl.rcParams["agg.path.chunksize"] = 0

N = 16
circle_radius = 0.9
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
initial_xs = circle_radius * np.cos(theta)
initial_ys = circle_radius * np.sin(theta)
initial_heading = theta + np.pi
initial_conditions = np.array([initial_xs, initial_ys, initial_heading])

r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    sim_in_real_time=False,
    initial_conditions=initial_conditions
)

goal_points = np.array([initial_xs, initial_ys, theta])
goal_points[0, :] = -initial_conditions[0, :]
goal_points[1, :] = -initial_conditions[1, :]

cmap = plt.get_cmap("tab20")
CM = cmap(np.linspace(0, 1, N))

radius = 0.25
a = 0.25
b = 0.25*0.8
w = 0.40

safety_radius_marker_size = determine_marker_size(r, radius)
font_height_points = determine_font_size(r, 0.2)
obs_r_marker_size = determine_marker_size(r, 0.10 * 10000)

H = init_hvis(r.axes, N, CM, radius=radius, a=a, b=b, w=w, grid_res=201, line_w=2)

circular_barrier = SICircularBarrierCertificate(
    barrier_gain=100,
    safety_radius=radius,
    magnitude_limit=0.2
)
barrier = SIEllipticalBarrierCertificate(
    barrier_gain=100,
    safety_a=a,
    safety_b=b,
    magnitude_limit=0.2
)
# barrier = SITriangleBarrierCertificate(
#     barrier_gain=100,
#     magnitude_limit=0.2
# )
# barrier = SISquareBarrierCertificate(
#     barrier_gain=100,
#     safety_width=w,
#     magnitude_limit=0.2
# )
# barrier = SIDeltaBarrierCertificate(
#     barrier_gain=100,
#     magnitude_limit=0.2,
#     barriers=[circular_barrier, elliptical_barrier, triangular_barrier, square_barrier],
# )

# Dynamics Transformation
si_to_uni, uni_to_si = create_si_to_uni_mapping()
# SI Controller
controller = create_si_position_controller()

# Create Goal Point Markers
goal_marker_size_m = 0.1
font_size = determine_font_size(r, 0.1)
line_width = 5
marker_size_goal = determine_marker_size(r, goal_marker_size_m)

# Text with goal identification
goal_caption = [f"G{ii}" for ii in range(goal_points.shape[1])]

# Plot text for caption
goal_points_text = [r.axes.text(goal_points[0,ii], goal_points[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-3)
for ii in range(goal_points.shape[1])]
goal_markers = [r.axes.scatter(goal_points[0,ii], goal_points[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-3)
for ii in range(goal_points.shape[1])]

trajectories = [[] for i in range(N)]

t = 0
start_time = None
dt = None
prev_time = None

x = r.get_poses()
thetas = x[2, :]
L = 0.05
r.step()

h_values = []
iterations = 0
negative_iterations = 0
complete = False
# Maximum of 4 minutes of time
max_iterations = 30 * 60 * 4
while iterations < max_iterations:
    start_time = time.time()
    x = r.get_poses()

    for i in range(N):
        trajectories[i].append(x[:, i].copy())

    x_si = uni_to_si(x)
    dxi = controller(x_si, goal_points[:2, :])
    dxi, h = barrier.apply(dxi, x_si)
    dxu = si_to_uni(dxi, x)

    if h < 0:
        negative_iterations += 1
    h_values.append(h)

    update_hvis(H, x, thetas, L, 3, 3, 1,
                    plot_scale=0.45, densify=True, densify_factor=150.0)

    r.set_velocities(np.arange(N), dxu)

    if np.linalg.norm(goal_points[:2, :] - x_si) < 0.08:
        complete = True
        break

    r.step()
    iterations += 1

if not complete:
    print("Simulation Did not Complete")
else:
    print(f"Simulation took {iterations * r.time_step} seconds")
print(f"H was negative in {negative_iterations} iterations")
r.call_at_scripts_end()

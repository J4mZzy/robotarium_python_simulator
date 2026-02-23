#
# Robots moving to random goals
#

import warnings
warnings.filterwarnings("ignore", module="matplotlib")

import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.misc import determine_marker_size, determine_font_size
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.transformations import create_si_to_uni_mapping

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os

from barrier_certificates import (
    SICircularBarrierCertificate,
    SIEllipticalBarrierCertificate,
    SISquareBarrierCertificate,
    SIDeltaBarrierCertificate,
    Obstacle
)
from graphing_utilities import init_hvis, update_hvis

def row_x(n, x_left, x_right):
    return np.linspace(x_left, x_right, n) if n > 0 else np.array([])

def generate_initial_conditions(N):
    # Layout params
    rect_width   = 2.8
    if N == 20:
        rect_width = 3.1
    rect_height  = 1.6
    margin_x     = 0.2        
    two_row_thresh = 8          # switch to two rows when N >= 8
    row_gap      = 0.4         # vertical gap between rows on a side

    x_left  = -rect_width/2  + margin_x
    x_right =  rect_width/2  - margin_x

    # ----- INITIAL CONDITIONS: line up at the BOTTOM, facing UP (+y) -----
    if N < two_row_thresh:
        # Single row at the bottom
        initial_x = row_x(N, x_left, x_right)
        initial_y = np.full(N, -rect_height/2)
        initial_heading = np.full(N, np.pi/2)     # face up

        initial_conditions = np.vstack([initial_x, initial_y, initial_heading])

        # GOALS: single row at the top, facing DOWN (-y)
        goals_top = np.vstack([
            row_x(N, x_left, x_right),
            np.full(N,  rect_height/2),
            np.full(N, -np.pi/2)                  # face down
        ])
    else:
        # Two rows per side
        n0 = (N + 1) // 2   # first row count
        n1 = N // 2         # second row count

        # Bottom rows (initial positions), both face UP
        yB0 = -rect_height/2
        yB1 = yB0 + row_gap

        initial_x = np.concatenate([row_x(n0, x_left, x_right), row_x(n1, x_left, x_right)])
        initial_y = np.concatenate([np.full(n0, yB0), np.full(n1, yB1)])-0.1
        initial_heading = np.full(N, np.pi/2)

        initial_conditions = np.vstack([initial_x, initial_y, initial_heading])

        # Top rows (goals), both face down
        yT0 =  rect_height/2 + 0.1
        yT1 =  yT0 - row_gap

        goals_top_x = np.concatenate([row_x(n0, x_left, x_right), row_x(n1, x_left, x_right)])
        goals_top_y = np.concatenate([np.full(n0, yT0), np.full(n1, yT1)])
        goals_top = np.vstack([goals_top_x, goals_top_y, np.full(N, -np.pi/2)])

    return initial_conditions, goals_top

import matplotlib as mpl
mpl.rcParams["path.simplify"] = False
mpl.rcParams["agg.path.chunksize"] = 0

obj_r = 0.16
o1 = Obstacle(np.array([-0.6, 0]), obj_r)
o2 = Obstacle(np.array([0.6, 0]), obj_r)

radius = 0.175
a = 0.2
b = 0.15
w = 0.175

circular_barrier = SICircularBarrierCertificate(
    safety_radius=radius,
    obstacles=[o1, o2]
)
elliptical_barrier = SIEllipticalBarrierCertificate(
    safety_a=0.2,
    safety_b=0.15,
    obstacles=[o1, o2]
)
vert_barrier = SIEllipticalBarrierCertificate(
    safety_a=0.15,
    safety_b=0.2,
    obstacles=[o1, o2]
)
square_barrier = SISquareBarrierCertificate(
    safety_width=0.175,
    obstacles=[o1, o2]
)
l2_barrier = SIDeltaBarrierCertificate(
    barriers=[
        square_barrier,
        vert_barrier,
    ],
    obstacles=[o1, o2]
)
l3_barrier = SIDeltaBarrierCertificate(
    barriers=[
        square_barrier,
        vert_barrier,
        circular_barrier
    ],
    obstacles=[o1, o2]
)
l4_barrier = SIDeltaBarrierCertificate(
    barriers=[
        square_barrier,
        vert_barrier,
        circular_barrier,
        elliptical_barrier
    ],
    obstacles=[o1, o2]
)
ellipses = SIDeltaBarrierCertificate(
    barriers=[
        elliptical_barrier,
        vert_barrier
    ],
    obstacles=[o1, o2]
)

barrier = vert_barrier
N = 8
seed = 1

os.makedirs(f"./data/{barrier.name()}/{seed}", exist_ok=True)

print(f"Running with {barrier.name()} at N={N}, Seed={seed}...")

rng = np.random.default_rng(seed)
initial_conditions, goal_points = generate_initial_conditions(N)
perm = rng.permutation(N)
goal_points = goal_points[:, perm]

r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    enable_safety=False,
    sim_in_real_time=True,
    drive_forward=False,
    initial_conditions=initial_conditions
)

cmap = plt.get_cmap("tab20")
CM = cmap(np.linspace(0, 1, N))

safety_radius_marker_size = determine_marker_size(r, radius)
font_height_points = determine_font_size(r, 0.2)

H = init_hvis(r.axes, N, CM, radius=radius*2, a=a*2, b=b*2, w=2*w, grid_res=201, line_w=2)

obj_r_marker_size = determine_marker_size(r, obj_r)
r.axes.scatter(o1.position[0], o1.position[1], s=obj_r_marker_size, marker='o', facecolors=[1, 0, 0], linewidth=5, zorder=-3)
r.axes.scatter(o2.position[0], o2.position[1], s=obj_r_marker_size, marker='o', facecolors=[0, 0, 1], linewidth=5, zorder=-3)

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

trajectories = [[] for _ in range(N)]
si_trajectories = [[] for _ in range(N)]

t = 0
dt = None
prev_time = None

x = r.get_poses()
circles = [patches.Circle((x[0, i], x[1, i]), radius=0.1, color="red") for i in range(N)]
for circle in circles:
    r.axes.add_patch(circle)
    circle.set_visible(False)

position_circles = [patches.Circle((x[0, i], x[1, i]), radius=0.001, color="blue") for i in range(N)]
for circle in position_circles:
    r.axes.add_patch(circle)
    circle.set_visible(True)
thetas = x[2, :]
L = 0.05
r.step()

times = []
h_values = []
u_values = []
simulation_weights = []
iterations = 0
negative_iterations = 0
complete = False
# Maximum of 4 minutes of time
max_iterations = 30 * 60 * 4
start_time = time.time()
zeros_iterations = 0
while time.time() - start_time < 60 * 4:
    x = r.get_poses()
    times.append(time.time() - start_time)
    for i in range(N):
        trajectories[i].append(x[:, i].copy())

    x_si = uni_to_si(x)
    for i in range(N):
        position_circles[i].center = (x_si[0, i], x_si[1, i])
        si_trajectories[i].append(x[:, i].copy())
    dxi = controller(x_si, goal_points[:2, :])
    dxi, h, negative_idxs = barrier.apply(dxi, x_si, x[2, :])
    u_norm = np.sum(np.linalg.norm(dxi, axis=0)) / N
    u_values.append(u_norm)
    if u_norm < 1e-3:
        zeros_iterations += 1
    else:
        zeros_iterations = 0
    # If the control inputs are near zero for 20 consecutive seconds, we can assume we have reached a deadlock
    if zeros_iterations >= 20 * 30:
        print("Deadlock")
        break
    dxu = si_to_uni(dxi, x)

    for i in range(N):
        if i in negative_idxs:
            circles[i].set_visible(True)
            circles[i].center = (x_si[0, i], x_si[1, i])
        else:
            circles[i].set_visible(False)

    if h < 0:
        negative_iterations += 1

    h_values.append(h)

    if isinstance(barrier, SIDeltaBarrierCertificate):
        update_hvis(H, x, thetas, L, barrier.current_shape(), barrier.target_shape(), barrier.delta_func(barrier.current_elapsed_time),
                        plot_scale=0.45, densify=True, densify_factor=150.0)
    else:
        
        update_hvis(H, x, thetas, L, barrier.current_shape(), barrier.target_shape(), 1,
                        plot_scale=0.45, densify=True, densify_factor=150.0)
        
    r.set_velocities(np.arange(N), dxu)

    if np.linalg.norm(goal_points[:2, :] - x_si) < 0.08:
        complete = True
        break

    r.step()
    simulation_weights.append(barrier.current_barrier_weight())
    iterations += 1

if complete:
    elapsed_time = time.time() - start_time
    print(elapsed_time)
else:
    print(0.0)
np.save(f"./data/{barrier.name()}/{seed}/trajectories", trajectories)
np.save(f"./data/{barrier.name()}/{seed}/h_min", np.array(h_values))
print(f"Negative Iterations: {negative_iterations}")
print(f"Max Violating: {np.min(h_values)}")
np.save(f"./data/{barrier.name()}/{seed}/u_avgs", np.array(u_values))
np.save(f"./data/{barrier.name()}/{seed}/barrier_weights", np.array(simulation_weights))
plt.close('all')

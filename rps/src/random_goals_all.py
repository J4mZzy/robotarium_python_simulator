#
# Robots moving to random goals
#

import rps.robotarium as robotarium
from rps.utilities.transformations import create_si_to_uni_mapping
from rps.utilities.misc import determine_marker_size, determine_font_size
from rps.utilities.controllers import create_si_position_controller
from rps.utilities.transformations import create_si_to_uni_mapping

import numpy as np
import matplotlib.pyplot as plt
import time
import polars as pl
import itertools

from barrier_certificates import (
    BarrierCertificate,
    SICircularBarrierCertificate,
    SIEllipticalBarrierCertificate,
    SISquareBarrierCertificate,
    SITriangleBarrierCertificate,
    SIDeltaBarrierCertificate,
    Obstacle
)
from graphing_utilities import init_hvis, update_hvis

def row_x(n, x_left, x_right):
    return np.linspace(x_left, x_right, n) if n > 0 else np.array([])

def generate_initial_conditions(N: int) -> tuple[np.ndarray, np.ndarray]:
    # Layout params
    rect_width   = 2.8
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

radius = 0.22
a = 0.25
b = 0.25*0.8
w = 0.40

circular_barrier = SICircularBarrierCertificate(
    barrier_gain=100,
    safety_radius=radius,
    magnitude_limit=0.2,
    obstacle_gain=10,
    obstacles=[o1, o2]
)
elliptical_barrier = SIEllipticalBarrierCertificate(
    barrier_gain=100,
    safety_a=a,
    safety_b=b,
    magnitude_limit=0.2,
    obstacle_gain=10,
    obstacles=[o1, o2]
)
square_barrier = SISquareBarrierCertificate(
    barrier_gain=100,
    safety_width=w,
    magnitude_limit=0.2,
    obstacle_gain=10,
    obstacles=[o1, o2]
)
triangle_barrier = SITriangleBarrierCertificate(
    barrier_gain=100,
    magnitude_limit=0.2,
    obstacle_gain=10,
    obstacles=[o1, o2]
)

barrier_library: list[BarrierCertificate] = []
for r in range(1, 5):
    for combination in itertools.combinations([circular_barrier, elliptical_barrier, square_barrier, triangle_barrier], r):
        if len(combination) > 1:
            barrier_library.append(SIDeltaBarrierCertificate(
                barrier_gain=100,
                magnitude_limit=0.2,
                barriers=list(combination),
                obstacle_gain=10,
                obstacles=[o1, o2]
            ))
        else:
            barrier_library.append(combination[0])

print("BARRIER LIBRARY")
for barrier in barrier_library:
    print(barrier.name())
print("----------------")

names = []
ns = []
seeds = []
completion_times = []
negative_hs = []
circle_weights = []
ellipse_weights = []
triangle_weights = []
square_weights = []

for barrier in barrier_library:
    for N in [4, 8, 11, 16, 20]:
        for seed in range(1, 6):
            print(f"...RUNNING {barrier.name()} at N={N} (Seed {seed})...")
            rng = np.random.default_rng(seed)
            initial_conditions, goal_points = generate_initial_conditions(N)
            perm = rng.permutation(N)
            goal_points = goal_points[:, perm]

            r = robotarium.Robotarium(
                number_of_robots=N,
                show_figure=False,
                sim_in_real_time=False,
                initial_conditions=initial_conditions
            )

            cmap = plt.get_cmap("tab20")
            CM = cmap(np.linspace(0, 1, N))

            safety_radius_marker_size = determine_marker_size(r, radius)
            font_height_points = determine_font_size(r, 0.2)

            H = init_hvis(r.axes, N, CM, radius=radius, a=a, b=b, w=w, grid_res=201, line_w=2)

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
            simulation_weights = []
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

                if isinstance(barrier, SIDeltaBarrierCertificate):
                    update_hvis(H, x, thetas, L, 3, 3, barrier.dleta_func(),
                                    plot_scale=0.45, densify=True, densify_factor=150.0)
                else:
                    update_hvis(H, x, thetas, L, 3, 3, 1,
                                    plot_scale=0.45, densify=True, densify_factor=150.0)


                r.set_velocities(np.arange(N), dxu)

                if np.linalg.norm(goal_points[:2, :] - x_si) < 0.08:
                    complete = True
                    break

                r.step()
                simulation_weights.append(barrier.current_barrier_weight())
                iterations += 1

            names.append(barrier.name())
            ns.append(N)
            seeds.append(seed)
            if not complete:
                completion_times.append(0.0)
                print("Simulation Did not Complete")
            else:
                completion_times.append(iterations * r.time_step)
                print(f"Simulation took {iterations * r.time_step} seconds")
            negative_hs.append(negative_iterations)
            simulation_weights = np.array(simulation_weights)
            weights = [np.sum(simulation_weights[:, i]) / iterations for i in range(len(simulation_weights[0]))]
            circle_weights.append(weights[0])
            ellipse_weights.append(weights[1])
            triangle_weights.append(weights[2])
            square_weights.append(weights[3])
            print(f"H was negative in {negative_iterations} iterations")
            r.call_at_scripts_end()
            plt.close('all')

print("...SAVING DATA...")
df = pl.DataFrame({
    "Barrier Name": names,
    "N": ns,
    "Seed": seeds,
    "Completion Time (s)": completion_times,
    "Negative Iterations": negative_hs,
    "Circle Time (%)": circle_weights,
    "Ellipse Time (%)": ellipse_weights,
    "Triangle Time (%)": triangle_weights,
    "Square Time (%)": square_weights
})
df.write_csv("./data/sim/random_goals.csv")
print("...DONE")
import rps.robotarium as robotarium
from rps.utilities.transformations import *
from barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time

import matplotlib as mpl
mpl.rcParams['path.simplify'] = False
mpl.rcParams['agg.path.chunksize'] = 0

'''This code is for the multi-robot circluar swapping simulations/experiments'''

# The robots will never reach their goal points so set iteration number (not used here, but if deadlock happens we can make the program stop in short time)
iterations = 1000

## The arena is bounded between x \in (-1.6,1.6)  y\in (-1,1) 

# Number of robots
N = 16 # 4,8,11,16,20

# radius of the circle robots are forming
circle_radius = 0.9

# Calculate initial positions in a circular formation
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)  # Angles for each robot
initial_x = circle_radius * np.cos(theta)  # X coordinates
initial_y = circle_radius * np.sin(theta)  # Y coordinates

# Headings (facing inward)
initial_heading = theta + np.pi  # Heading towards the center (add pi to point inward)

# Combine initial positions into the required format (x, y, theta)
initial_conditions = np.array([initial_x, initial_y, initial_heading])

# Instantiate Robotarium object
r = robotarium.Robotarium(
    number_of_robots=N,
    show_figure=True,
    sim_in_real_time=True,
    initial_conditions=initial_conditions
)

# Define goal points: for a swapping behavior, we can simply offset the current positions
goal_points = np.array([initial_x, initial_y, theta])  # Start by setting goal points to the current positions
goal_points[0, :] = -initial_conditions[0,:] 
goal_points[1, :] = -initial_conditions[1,:]

## Plotting Parameters

#GT color scheme
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
safety_radius_view = 0.2
safety_radius_marker_size = determine_marker_size(r,safety_radius_view) # Will scale the plotted markers to be the diameter of provided argument (in meters)
font_height_meters = 0.2
font_height_points = determine_font_size(r,font_height_meters) # Will scale the plotted font height to that of the provided argument (in meters)
obs_r = 0.10*10000 # for visual
obs_r_marker_size = determine_marker_size(r,obs_r) # Will scale the plotted markers to be the diameter of provided argument (in meters)

# Create single integrator position controller
si_position_controller = create_si_position_controller()

# Initialize parameters
radius = 0.25
a = 0.25
b = 0.25*0.8
w = 0.40
base_shape = 1 # default CBF shape

############################# Plotting helper ######################
def lse3(L1, L2, L3):
    """Stable log-sum-exp for arrays."""
    M = np.maximum.reduce([L1, L2, L3])
    return M + np.log(np.exp(L1 - M) + np.exp(L2 - M) + np.exp(L3 - M))

def densify_segments(segs, max_step):
    """Insert points so edges are <= max_step; segs: list of (m x 2) arrays."""
    out = []
    for seg in segs:
        if len(seg) < 2:
            continue
        pts = [seg[0]]
        for a, b in zip(seg[:-1], seg[1:]):
            d = float(np.hypot(*(b - a)))
            n = int(np.ceil(d / max_step))
            if n <= 1:
                pts.append(b)
            else:
                t = np.linspace(0.0, 1.0, n+1)[1:]
                pts.extend(a + (b - a) * t[:, None])
        out.append(np.asarray(pts))
    return out

# Try to use the internal (fast) contour generator
try:
    from matplotlib import _contour
    HAS_CONTOURGEN = True
except Exception:
    HAS_CONTOURGEN = False

def init_hvis(ax, N, CM, *, radius, a, b, w, p=3, Rvis_scale=1.4, grid_res=201, line_w=2):
    """
    ax: matplotlib axes (e.g., r.axes), N: #agents, CM: (N x 3/4) colors
    returns a dict 'H' with precomputed fields and artists.
    """
    radius = float(radius); a = float(a); b = float(b); w = float(w)
    rt3 = np.sqrt(3.0)

    # local grid in (u,v) — agent frame
    Rvis = Rvis_scale * max(radius, a, b, w/2.0)
    nu = nv = int(grid_res)
    u_vals = np.linspace(-Rvis, Rvis, nu)
    v_vals = np.linspace(-Rvis, Rvis, nv)
    UU, VV = np.meshgrid(u_vals, v_vals)

    # fields that only change if radius/a/b/w/p change
    h_circ_l  = (UU/radius)**2 + (VV/radius)**2 - 1.0
    h_ellip_l = (UU/a)**2      + (VV/b)**2      - 1.0
    h_sq_l    = (np.abs(UU)**p + np.abs(VV)**p)**(1.0/p) - w/2.0
    L1 =  4*UU + 4*rt3*VV
    L2 = -8*UU
    L3 =  4*UU - 4*rt3*VV
    h_tri_l = (3.0/5.0) * lse3(L1, L2, L3) - 1.0
    h_fields_local = [h_circ_l, h_ellip_l, h_tri_l, h_sq_l]

    # one LineCollection per agent; add once
    lc_list = []
    for i in range(N):
        lc = LineCollection(
            [], colors=[CM[i]], linewidths=line_w, zorder=3,
            antialiased=True, capstyle='round', joinstyle='round'
        )
        ax.add_collection(lc)
        lc_list.append(lc)

    return {
        'ax': ax, 'N': N, 'CM': CM,
        'UU': UU, 'VV': VV, 'Rvis': Rvis,
        'h_fields_local': h_fields_local,
        'lc_list': lc_list
    }
# ---------- per-frame update ----------
def update_hvis(H, x, thetas, L, base_shape, target_shape, Delta, *, plot_scale=0.45,
                densify=True, densify_factor=150.0):
    """
    Updates the per-agent LineCollections to draw the 0-level of h_tv.
    base_shape, target_shape in {1,2,3,4}; Delta in [0,1].
    thetas in radians; x is 3xN (unicycle state).
    """
    UU, VV   = H['UU'], H['VV']
    h_local  = H['h_fields_local']
    lc_list  = H['lc_list']
    ax       = H['ax']
    N        = H['N']
    Rvis     = H['Rvis']

    # blend ONCE in (u,v)
    h_base_l = h_local[base_shape  - 1]
    h_tgt_l  = h_local[target_shape - 1]
    h_tv_l   = (1.0 - Delta) * h_base_l + Delta * h_tgt_l

    # get the 0-level segments w/o leaving artists in the axes
    if HAS_CONTOURGEN:
        cg = _contour.QuadContourGenerator(UU, VV, h_tv_l, None, True, 0)
        segs_local = cg.create_contour(0.0)   # list of (m x 2) in (u,v)
    else:
        cs = ax.contour(UU, VV, h_tv_l, levels=[0], linewidths=0)
        segs_local = cs.allsegs[0] if getattr(cs, "allsegs", None) else []
        try: cs.remove()
        except Exception: pass

    # optional smoothing of polylines
    if densify and len(segs_local):
        segs_local = densify_segments(segs_local, max_step=Rvis / densify_factor)

    # update each agent by rigidly transforming local segs
    for i in range(N):
        cx = x[0, i] + L*np.cos(x[2, i])
        cy = x[1, i] + L*np.sin(x[2, i])
        c  = np.cos(thetas[i]); s = np.sin(thetas[i])

        # world coords: other = (cx,cy) - R(theta_i) @ (scaled [u;v])
        if len(segs_local) == 0:
            lc_list[i].set_segments([])
            continue

        segs_world = [
            np.column_stack((
                cx + (c*(plot_scale*seg[:,0]) - s*(plot_scale*seg[:,1])),
                cy + (s*(plot_scale*seg[:,0]) + c*(plot_scale*seg[:,1]))
            ))
            for seg in segs_local
        ]
        lc_list[i].set_segments(segs_world)

H = init_hvis(r.axes, N, CM, radius=radius, a=a, b=b, w=w, grid_res=201, line_w=2)

############################################ CBF Library #######################################################
# We're working in single-integrator dynamics, and we don't want the robots
# to collide.  Thus, we're going to use barrier certificates (in a centrialized way)
CBF_n = 4 # how many CBFs we are using 

si_barrier_cert_cir = create_single_integrator_barrier_certificate(barrier_gain=100,safety_radius=radius)
si_barrier_cert_ellip = create_single_integrator_barrier_certificate_ellipse(barrier_gain=100,safety_a=a,safety_b=b)
# si_barrier_cert_tri = create_single_integrator_barrier_certificate_triangle_with_obstacles(barrier_gain=1)
# si_barrier_cert_sqaure = create_single_integrator_barrier_certificate_square_with_obstacles(barrier_gain=1,safety_width=w,norm=3)

######## Remember to change this to 1 when running ellipse ######################
current_target_shape = 1 # initialize the shape flag as 1 (1 is circle and 2 is ellipse)

# Default shape (begin with circle)         
lamb = np.array([1.0, 0.0, 0.0, 0.0]) # current lambda array (storing the current shape)
Delta = 0 # Delta
lamb_list = []
Delta_list = []
target_list = []

# Initialize the transition variables
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

#Arrow for desired orientation (optional)
# goal_orientation_arrows = [r.axes.arrow(goal_points[0,ii], goal_points[1,ii], goal_marker_size_m*np.cos(goal_points[2,ii]), goal_marker_size_m*np.sin(goal_points[2,ii]), width = 0.01, length_includes_head=True, color = CM[ii,:], zorder=-2)
# for ii in range(goal_points.shape[1])]

# Plot text for caption
goal_points_text = [r.axes.text(goal_points[0,ii], goal_points[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-3)
for ii in range(goal_points.shape[1])]
goal_markers = [r.axes.scatter(goal_points[0,ii], goal_points[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-3)
for ii in range(goal_points.shape[1])]

## Obstacles 
obstacle_1 = r.axes.scatter(0.3, 0.2, s=obs_r, marker='o', facecolors=[1, 0, 0],linewidth=line_width,zorder=-3) # [133/255, 116/255, 55/255]
obstacle_1 = r.axes.scatter(-0.3, -0.2, s=obs_r, marker='o', facecolors=[0, 0, 1],linewidth=line_width,zorder=-3) # [133/255, 116/255, 55/255]

r.step()
# r.step_no_error() (for robotarium with 20 agents)

# Initialize a list to keep track of the scatter objects
g_objects = []

# Initialize a list to keep track of the robots' trajectories
trajectories = {i: [] for i in range(N)}

# Initialize empty lists to store norm values
norm_dxi_cir_list = []
norm_dxi_ellip_list = []
norm_dxi_tv_list = []

## initialize for loop speed calculation
t = 0 # initialize time for Delta 
start_time = None # timer variable in loop
dt = None
prev_time = None

# While the goal is not reached
while(1):
# for iter in range(iterations):
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

        ########################### barrier type ######################################
        # Use the barrier certificates to make sure that the agents don't collide
        # Generating safe inputs
        dxi_cir, h_min_cir = si_barrier_cert_cir(dxi, x_si)                # the first barrier being circular
        dxi_ellip, h_min_ellip = si_barrier_cert_ellip(dxi, x_si,thetas)     # the second barrier being elliptical

        ############################# selection ########################################
        # Use the second single-integrator-to-unicycle mapping to map to unicycle
        dxu_cir = si_to_uni_dyn(dxi_cir, x) # circular
        dxu_ellip = si_to_uni_dyn(dxi_ellip, x) # elliptical
        
        # Default shape
        # For target shape, circle =1, ellipse=2
        norm_dxi_cir = np.linalg.norm(dxi_cir,ord=2)
        norm_dxi_ellip = np.linalg.norm(dxi_ellip,ord=2)
        ##TODO find the norm of the actual input in si

        # Append the norms to the lists for post-processing
        norm_dxi_cir_list.append(norm_dxi_cir)
        norm_dxi_ellip_list.append(norm_dxi_ellip)

        # Finding s_t, which is the shape we are morphing to
        # desired_target_shape = np.argmax([norm_dxi_cir,norm_dxi_ellip]) + 1 # s_t (shape to morph into) (1 is circle, 2 is ellipse)
        sorted_target_shapes = np.argsort([-norm_dxi_cir,-norm_dxi_ellip]) # sorted list
        # print("index:",desired_target_shape)

        # h_min from each certificate
        hmins = np.array([h_min_cir, h_min_ellip], dtype=float)

        for i in range(2): # just 2 here 
            if hmins[sorted_target_shapes[i]] > 0:
                desired_target_shape = sorted_target_shapes[i] + 1 
                break

        ######################################################################################
        if prev_time is None:
            prev_time = time.time() # start timer
        now = time.time() # start counter
        dt = now - prev_time 

        prev_time = time.time() # record time
        ##########################################################################################
        ## If target has changed, reset Delta and update lambda

        # print("current_target_shape",current_target_shape)
        
        ## Delta = (1-cos(pi*t))/2 for t \in [0,1), and = 1 if t >= 1
        ## switch if shape has been reached, and set lambda to 1, if target no reached then don't switch
        if 0 <= t < 1*2:
            Delta = np.clip(Delta + np.pi/2/2*np.sin(np.pi/2*t)*dt, 0, 1)  # update Delta   
        else:
            Delta = 1 
        # calculate Delta dot 
        if Delta < 1:
            Delta_dot = np.pi/2/2*np.sin(np.pi/2*t) # compute delta dot
            # t =+ dt # update time
        else:
            Delta_dot = 0 #transformation complete
            Delta = 1
        # print(Delta_dot)
        t = t + dt # update time
        if current_target_shape != desired_target_shape:
            if Delta == 1: # completed transformation to another shape and another target is selected        
                for i in range(CBF_n):
                    if i == current_target_shape - 1:
                        lamb[i] = 1  # (1-Delta)*lamb[i] + Delta 
                        base_shape = i+1 # keeps track of the base shape (lambda index + 1)
                    else: 
                        # not the target shape, set to 0
                        lamb[i] = 0   # (1-Delta)*lamb[i] 
                current_target_shape = desired_target_shape # switch target shape
                Delta = 0 # reset Delta
                t = 0 # reset time                  


        # print(t)
        # print("Delta",Delta_dot)
        # print("lambda",lamb)

        # if iter >= 205:
        #     time.sleep(3)

        # print("iterations:", iter)
        lamb_list.append(lamb.copy())
        Delta_list.append(Delta)
        target_list.append(current_target_shape)
        ########################################Time varying CBF#####################################
        si_barrier_cert_tv = create_single_integrator_barrier_certificate_time_varying(Delta=Delta,lamb=lamb,target_shape=current_target_shape,Delta_dot=Delta_dot
                                                                                       ,barrier_gain=100,safety_radius=radius
                                                                                       ,safety_a=a,safety_b=b)  

        # si_barrier_cert_tv = create_single_integrator_barrier_certificate_ellipse(barrier_gain=1,safety_a=a,safety_b=b)
        dxi_tv = si_barrier_cert_tv(dxi, x_si, thetas)  
        dxu_tv = si_to_uni_dyn(dxi_tv, x)      
        dxu = dxu_tv

        # dxu = dxu_ellip # for invariant-CBF experiments 

        # Append the norms to the lists for post-processing
        norm_dxi_tv = np.linalg.norm(dxi_tv,ord=2)
        norm_dxi_tv_list.append(norm_dxi_tv)

        # Remove previous scatter plot markers
        for g in g_objects:
            g.remove()

        # Clear the list after removing markers
        g_objects = []  

        ############################# for plotting only #######################################   
        update_hvis(H, x, thetas, L, base_shape, current_target_shape, Delta,
                    plot_scale=0.45, densify=True, densify_factor=150.0)

        #######################################################################################

        # Set the velocities by mapping the single-integrator inputs to unciycle inputs
        r.set_velocities(np.arange(N), dxu)

        # Stopping cirterion (if goal is reached)
        if(np.linalg.norm(goal_points[:2,:] - x_si) < 0.08):
            break # navigation completed

        # Iterate the simulation
        r.step()
        # r.step_no_error() (for robotarium with 20 agents)

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()


# Convert lists to a single 2D NumPy array
u_norms_array = np.column_stack((norm_dxi_cir_list, norm_dxi_ellip_list,norm_dxi_tv_list))

#Save Data
print(time.time() - exp_start_time)
np.save('trajectories', trajectories)
np.save("u_norms", u_norms_array)
np.save("lamb_list", lamb_list)
np.save("Delta_list", Delta_list)
np.save("target_list", target_list)

######################## plot block ############################################################

## Plotting the position trajectories
# print("Preparing to plot trajectories...")
## Set the font globally to Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
## Enable LaTeX plotting
# plt.rc('text', usetex=True)

# plt.figure(figsize=(8, 8))
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


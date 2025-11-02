import rps.robotarium as robotarium
from rps.utilities.transformations import *
from barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import time

'''This code is for the multi-robot circluar swapping simulations/experiments'''

# The robots will never reach their goal points so set iteration number (not used here, but if deadlock happens we can make the program stop in short time)
iterations = 1000

## The arena is bounded between x \in (-1.6,1.6)  y\in (-1,1) 

# Number of robots
N = 16 # 2,4,8,11,16,20

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
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True, initial_conditions=initial_conditions)
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
safety_radius_view = 0.15
safety_radius_marker_size = determine_marker_size(r,safety_radius_view) # Will scale the plotted markers to be the diameter of provided argument (in meters)
font_height_meters = 0.2
font_height_points = determine_font_size(r,font_height_meters) # Will scale the plotted font height to that of the provided argument (in meters)

# Create single integrator position controller
si_position_controller = create_si_position_controller()

# Initialize parameters
radius1 = 0.25
a1 = 0.25 
b1 = 0.25*0.95
radius2 = 0.25*0.95
a2 = 0.25*0.95 
b2 = 0.25*0.90
radius3 = 0.25*90
a3 = 0.25*0.90 
b3 = 0.25*0.85
radius4 = 0.25*85
a4 = 0.25*0.85 
b4 = 0.25*0.80
radius5 = 0.25*80
a5 = 0.25*0.80 
b5 = 0.25*0.75


############################################ CBF Library #######################################################
# We're working in single-integrator dynamics, and we don't want the robots
# to collide.  Thus, we're going to use barrier certificates (in a centrialized way)
# CBF_n = 10 # how many CBFs we are using 

si_barrier_cert_cir1 = create_single_integrator_barrier_certificate(barrier_gain=1,safety_radius=radius1)
si_barrier_cert_ellip1 = create_single_integrator_barrier_certificate_ellipse(barrier_gain=1,safety_a=a1,safety_b=b1)
si_barrier_cert_cir2 = create_single_integrator_barrier_certificate(barrier_gain=1,safety_radius=radius2)
si_barrier_cert_ellip2 = create_single_integrator_barrier_certificate_ellipse(barrier_gain=1,safety_a=a2,safety_b=b2)
si_barrier_cert_cir3 = create_single_integrator_barrier_certificate(barrier_gain=1,safety_radius=radius3)
si_barrier_cert_ellip3 = create_single_integrator_barrier_certificate_ellipse(barrier_gain=1,safety_a=a3,safety_b=b3)
si_barrier_cert_cir4 = create_single_integrator_barrier_certificate(barrier_gain=1,safety_radius=radius4)
si_barrier_cert_ellip4 = create_single_integrator_barrier_certificate_ellipse(barrier_gain=1,safety_a=a4,safety_b=b4)
si_barrier_cert_cir5 = create_single_integrator_barrier_certificate(barrier_gain=1,safety_radius=radius5)
si_barrier_cert_ellip5 = create_single_integrator_barrier_certificate_ellipse(barrier_gain=1,safety_a=a5,safety_b=b5)


t = 0 

cir_certs = [si_barrier_cert_cir1, si_barrier_cert_cir2, si_barrier_cert_cir3,
             si_barrier_cert_cir4, si_barrier_cert_cir5]
ell_certs = [si_barrier_cert_ellip1, si_barrier_cert_ellip2, si_barrier_cert_ellip3,
             si_barrier_cert_ellip4, si_barrier_cert_ellip5]

# Interleave to get codes 1..10: 1=cir1, 2=ellip1, 3=cir2, 4=ellip2, ..., 9=cir5, 10=ellip5
certs = [c for pair in zip(cir_certs, ell_certs) for c in pair]
CBF_n = len(certs)  # 10

def code_to_kind_k(code_1b: int):
    """Return ('cir'|'ell'), k in 1..5, given 1-based code."""
    assert 1 <= code_1b <= 10
    if code_1b % 2 == 1:
        return 'cir', (code_1b + 1)//2
    else:
        return 'ell', code_1b//2

def allowed_targets(code_cur_1b: int):
    """Adjacency rule with 'stay' allowed to avoid thrash."""
    kind, k = code_to_kind_k(code_cur_1b)
    allowed = {code_cur_1b}  # always allow staying
    if kind == 'cir':
        allowed.add(2*k)                 # ellip k
        if k < 5: allowed.add(2*(k+1))   # ellip k+1
    else:  # kind == 'ell'
        allowed.add(2*k-1)               # cir k
        if k > 1: allowed.add(2*(k-1)-1) # cir k-1
    return sorted(allowed)

def apply_cert_by_code(code_1b: int, dxi_si: np.ndarray, x_si: np.ndarray, thetas: np.ndarray):
    """Call the correct certificate; ellipse needs thetas, circle does not."""
    cert = certs[code_1b - 1]
    if code_1b % 2 == 0:   # ellipse
        return cert(dxi_si, x_si, thetas)
    else:                  # circle
        return cert(dxi_si, x_si)

#################################################################################################################

######## Remember to change this to 1 when running ellipse ######################
previous_target_shape = 1 # initialize the shape flag as 1 (1 is circle and 2 is ellipse)
target_array = np.zeros(CBF_n)
target_array[previous_target_shape-1] = 1

# Default shape (begin with circle)         
Delta_cur = np.array([1.0,0.0]) # current Delta array
lamb = np.array([1.0,0.0,0.0,0.0]) # current lambda array (storing the current shape)
Delta = 0 # Delta

lamb_list = []
Delta_list = []
target_list = []

# Initialize the transition variables
transition_in_progress = False
# start_time = None
T = np.pi/2  # Duration for the morphing transition in seconds
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

#Plot text for caption
goal_points_text = [r.axes.text(goal_points[0,ii], goal_points[1,ii], goal_caption[ii], fontsize=font_size, color='k',fontweight='bold',horizontalalignment='center',verticalalignment='center',zorder=-3)
for ii in range(goal_points.shape[1])]
goal_markers = [r.axes.scatter(goal_points[0,ii], goal_points[1,ii], s=marker_size_goal, marker='s', facecolors='none',edgecolors=CM[ii,:],linewidth=line_width,zorder=-3)
for ii in range(goal_points.shape[1])]

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
start_time = None # timer variable in loop
dt = None
prev_time = None

# --- state for morphing ---
if 'previous_target_shape' not in globals(): previous_target_shape = 1   # start at cir1
if 'Delta' not in globals(): Delta = 1.0
if 't' not in globals(): t = 1.0
if 'prev_time' not in globals(): prev_time = None
lamb = np.zeros(CBF_n); lamb[previous_target_shape-1] = 1.0  # one-hot over 10 shapes
target_array = np.zeros(CBF_n)

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
    
        dxi_by_code = []
        norm_by_code = []
        for code in range(1, CBF_n+1):
            dxi_f = apply_cert_by_code(code, dxi, x_si, thetas)
            dxi_by_code.append(dxi_f)
            # use Frobenius norm (overall effort)
            norm_by_code.append(np.linalg.norm(dxi_f, ord='fro'))
    
        # Choose target shape constrained by adjacency
        allowed = allowed_targets(previous_target_shape)
        # Pick the allowed shape with maximal norm (matching your prior heuristic)
        target_shape = allowed[int(np.argmax([norm_by_code[c-1] for c in allowed]))]
    
        # One-hot target array
        target_array[:] = 0.0
        target_array[target_shape - 1] = 1.0
    
        # ---- time and Delta schedule (ease-in-out) ----
        now = time.perf_counter()
        dt = 0.0 if prev_time is None else (now - prev_time)
        prev_time = now
    
        # If target changed and previous morph finished, snap lambdas and reset
        if previous_target_shape != target_shape and Delta == 1.0:
            lamb[:] = 0.0
            lamb[target_shape - 1] = 1.0
            previous_target_shape = target_shape
            t = 0.0
            Delta = 0.0
    
        # Smooth progression: Delta(t) = (1 - cos(pi t))/2 on t ∈ [0,1], then hold
        if t < 1.0:
            Delta_dot = (np.pi/2.0) * np.sin(np.pi * t)
            Delta = min(1.0, Delta + Delta_dot * dt)
            t += dt
        else:
            Delta_dot = 0.0
            Delta = 1.0
    
    
            # print(t)
            # print("Delta",Delta)
            # print("lambda",lamb)
    
            # if iter >= 205:
        #     time.sleep(3)

        # print("iterations:", iter)
        lamb_list.append(lamb.copy())
        Delta_list.append(Delta)
        target_list.append(target_shape)

        ##########################################################################################
        # print("a_cur:",a_cur)
        # print("b_cur:",b_cur)

        ## Delta cur is used to get the current convex combination CBF, and Delta target is used to calculate h3 (time varying CBF)!

        # si_barrier_cert_tv, idx_sel, (w1_sel, w2_sel) = pick_cert_for_Delta(Delta_cur, target_shape)
        si_barrier_cert_tv = create_single_integrator_barrier_certificate_time_varying(Delta=Delta,lamb=lamb,target_shape=target_shape,t=t
                                                                                       ,barrier_gain=10,safety_radius=radius
                                                                                       ,safety_a=a,safety_b=b)  

        # si_barrier_cert_tv = create_single_integrator_barrier_certificate_ellipse(barrier_gain=1,safety_a=a,safety_b=b)
        dxi_tv = si_barrier_cert_tv(dxi, x_si, thetas)  
        dxu_tv = si_to_uni_dyn(dxi_tv, x)      
        dxu = dxu_tv
        # dxu = dxu_ellip

        norm_dxi_tv = np.linalg.norm(dxi_tv,ord=2)
        # Append the norms to the lists for post-processing
        norm_dxi_tv_list.append(norm_dxi_tv)

        ## Delta cur has ([circle,ellipse])
        # Delta_cur = Delta_target # update Delta current     
        # Remove previous scatter plot markers
        for g in g_objects:
            g.remove()

        # Clear the list after removing markers
        g_objects = []  

        ## Update Plotted Visualization
        # g.set_offsets(x[:2,:].T+np.array([L*np.cos(x[2,:]),L*np.sin(x[2,:])]).T)

        ## This updates the marker sizes if the figure window size is changed. 
        # g.set_sizes([determine_marker_size(r,safety_radius)])

        # g_objects.append(g)

        ############################# for plotting only #######################################
        # error = x[:] - x[:]
        # # circular CBF
        # h_circ = (error[0]*error[0] + error[1]*error[1]) - np.power(safety_radius, 2)
        # # ellipitical CBF
        # error_1 = (error[0]*np.cos(theta[j])+error[1]*np.sin(theta[j])) / safety_a
        # error_2 = (error[0]*np.sin(theta[j])-error[1]*np.cos(theta[j])) / safety_b 
        # h_ellip = error_1**2 + error_2**2 - 1 
        # # calculate the current h (convex combination)
        # h_prev = lamb[0] * h_circ +  lamb[1] * h_ellip 
        # if target_shape == 1:
        #         # calculate h_3 
        #         h_tv = (1-Delta) * h_prev + Delta * h_circ 
        # if target_shape == 1:
        
        #########################################################################################
        # Create and add ellipses to the axes
        # for i in range(N):
        #     ellipse = Ellipse(xy=(x[0, i] + L * np.cos(x[2, i]), x[1, i] + L * np.sin(x[2, i])),
        #                     width=a_cur*0.9, height=b_cur*0.9, angle=np.degrees(thetas[i]),
        #                     facecolor='none', edgecolor=CM[i], linewidth=2)
        #     r.axes.add_patch(ellipse)
        #     g_objects.append(ellipse)  # Keep track of the patches       

        # Set the velocities by mapping the single-integrator inputs to unciycle inputs
        r.set_velocities(np.arange(N), dxu)

        # Stopping cirterion (if goal is reached)
        if(np.linalg.norm(goal_points[:2,:] - x_si) < 0.08):
            break

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

## plot block

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


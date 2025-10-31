import numpy as np
import matplotlib.pyplot as plt

'''This code generates the plot for the minimum h (CBF output) value between all the robots during the experiment 
from reading the data saved through the simulations/experiments'''

# Load data
trajectories = np.load('trajectories.npy', allow_pickle=True).item()

lambs = np.load('lamb_list.npy')
# print(lambs)
Deltas = np.load('Delta_list.npy')
targets = np.load('target_list.npy')
# print(Deltas)


# Number of robots in the simulation/experiment
N = 4 # 2,4,8,11,16,20

# time steps
T = np.array(trajectories[0]).shape[0]
h_min = np.empty(T)
# print(np.array(trajectories[0])[1,:])

## Setting values for the barrier parameters
r = 0.25  # Circle radius
a = 0.25  # Ellipse major axis length
b = 0.2   # Ellipse minor axis length

for t in range(T):
    h_min[t] = np.inf
    for i in range(N):
        x_i = np.array(trajectories[i])[t,:]
        for j in range(N):
            if i == j:
                continue
            x_j = np.array(trajectories[j])[t,:]
            
            ## Circular CBF
            error_0 = x_i[0]-x_j[0]
            error_1 = x_i[1]-x_j[1]

            h_ij_cir = (error_0*error_0 + error_1*error_1) - np.power(r, 2)

            error_1 = (error_0*np.cos(x_i[2])+error_1*np.sin(x_i[2])) 
            error_2 = (error_0*np.sin(x_i[2])-error_1*np.cos(x_i[2]))
             
            ## Elliptical CBF
            h_ij_ell = (error_1/a)**2 + (error_2/b)**2 - 1 

            p = 3
            safety_width = 0.4

            h_ij_square = (np.power(np.power(np.abs(error_1),p) + np.power(np.abs(error_2),p),1/p)) - safety_width/2

            h_ij_tri = 3/5 * np.log(np.exp(4*error_1+4*np.sqrt(3)*error_2) + np.exp(-8*error_1) + np.exp(4*error_1-4*np.sqrt(3)*error_2))- 1


            # h_ij = np.min(np.array([h_ij_cir,h_ij_ell]))
            h_prev = lambs[t][0]*h_ij_cir + lambs[t][1]*h_ij_ell + lambs[t][2]*h_ij_tri + lambs[t][3]*h_ij_square

            ## 1 morph to circle, 2 morph to ellipse
            if targets[t] == 1:
                h_ij = (1-Deltas[t]) * h_prev + Deltas[t] * h_ij_cir 
            elif targets[t] == 2:
                h_ij = (1-Deltas[t]) * h_prev + Deltas[t] * h_ij_ell 
            elif targets[t] == 3:
                h_ij = (1-Deltas[t]) * h_prev + Deltas[t] * h_ij_tri 
            elif targets[t] == 4:
                h_ij = (1-Deltas[t]) * h_prev + Deltas[t] * h_ij_square 

    ## Pick the samller one 
    if h_ij < h_min[t]:
        h_min[t] = h_ij

## Variables
iterations = np.arange(T)
h_min_global = np.min(h_min)
index = np.argmin(h_min)
zeros = np.zeros(T)

## Display minimum value of all time on terminal
print(h_min_global)

## Plot block
plt.figure(figsize=(8, 5.5))

# Set global font to Times New Roman and enable LaTeX
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

# Plot with different line styles and thickness
plt.plot(iterations, h_min, linestyle='solid', linewidth=3, color='b')
# plt.plot(iterations, h_min, marker='o', linestyle='none', markersize=3, color='b')
plt.plot(iterations, zeros,
         linestyle='dashed', linewidth=3, color='r')
# plt.scatter(index, h_min_global, color='black', marker='.', s=400, label="global $h_{/min}$")


plt.xlabel(r"\textbf{Time step}", fontsize=22)
plt.ylabel(r"$\mathbf{h_{\mathbf{\min}}}$", fontsize=22)

# Adjust axis ticks for better readability
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# Display the plot
# plt.savefig("h_min_20_agents", dpi=600)
plt.show()




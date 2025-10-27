import numpy as np
import matplotlib.pyplot as plt

'''This code generates the plot for the minimum h (CBF output) value between all the robots during the experiment 
from reading the data saved through the simulations/experiments'''

# Load data
trajectories = np.load('trajectories.npy', allow_pickle=True).item()

# Number of robots in the simulation/experiment
N = 11 # 2,4,8,11,16,20

# time steps
T = np.array(trajectories[0]).shape[0]
h_min = np.empty(T)
# print(np.array(trajectories[0])[1,:])

## Setting values for the barrier parameters
r = 0.25 # Circle radius
a = 0.25 # Ellipse major axis length
b = 0.2  # Ellipse minor axis length

for t in range(T):
    h_min[t] = np.inf
    for i in range(N-1):
        x_i = np.array(trajectories[i])[t,:]
        for j in range(i+1, N):
            x_j = np.array(trajectories[j])[t,:]
            
            ## Circular CBF
            error_0 = x_i[0]-x_j[0]
            error_1 = x_i[1]-x_j[1]
            h_ij_cir = (error_0*error_0 + error_1*error_1) - np.power(r, 2)

            ## Elliptical CBF
            error_2 = (error_0*np.cos(x_i[2])+error_1*np.sin(x_i[2])) / a
            error_3 = (error_0*np.sin(x_i[2])-error_1*np.cos(x_i[2])) / b
            h_ij_ell = error_2**2 + error_3**2 - 1  

            h_ij = np.min(np.array([h_ij_cir,h_ij_ell]))

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
plt.plot(iterations, h_min, 
         linestyle='solid', linewidth=3, color='b')
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




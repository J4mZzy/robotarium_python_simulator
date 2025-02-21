import numpy as np
import matplotlib.pyplot as plt

# load data
trajectories = np.load('trajectories.npy', allow_pickle=True).item()

N = 20 # 2,4,8,11,16,20

# trajectory = np.array(trajectories[i]) 

# time steps
T = np.array(trajectories[0]).shape[0]
h_min = np.empty(T)
# print(np.array(trajectories[0])[1,:])
for t in range(T):
    h_min[t] = np.inf
    for i in range(N-1):
        x_i = np.array(trajectories[i])[t,:]
        for j in range(i+1, N):
            x_j = np.array(trajectories[j])[t,:]
            
            ## Circle
            error_0 = x_i[0]-x_j[0]
            error_1 = x_i[1]-x_j[1]
            h_ij_cir = (error_0*error_0 + error_1*error_1) - np.power(0.25, 2)

            # ellipse
            error_3 = (error_0*np.cos(x_i[2])+error_1*np.sin(x_i[2])) / 0.25
            error_4 = (error_0*np.sin(x_i[2])-error_1*np.cos(x_i[2])) / 0.2
            h_ij_ell = error_3**2 + error_4**2 - 1  

            h_ij = np.min(np.array([h_ij_cir,h_ij_ell]))

    if h_ij < h_min[t]:
        h_min[t] = h_ij

iterations = np.arange(T)
h_min_global = np.min(h_min)
index = np.argmin(h_min)
zeros = np.zeros(T)
print(h_min_global)

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



# we only need to find the error produced by the closest agents

# error_1 = (error[0]*np.cos(theta[agent_index])+error[1]*np.sin(theta[agent_index])) / safety_a
# error_2 = (error[0]*np.sin(theta[agent_index])-error[1]*np.cos(theta[agent_index])) / safety_b

# h = error_1**2 + error_2**2 - 1   



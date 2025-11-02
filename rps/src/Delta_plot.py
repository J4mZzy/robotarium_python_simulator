import numpy as np
import matplotlib.pyplot as plt
# Load data
Deltas = np.load('Delta_list.npy')
lambs = np.load("lamb_list.npy")
print(lambs.shape)

T = lambs.shape[0]
index = np.zeros(T)

for t in range(T):
    for i in range (lambs.shape[1]):
        if lambs[t][i] == 1:
            index[t] = i

# Set global font to Times New Roman and enable LaTeX
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

# Plot with different line styles and thickness
iterations = np.arange(T)
plt.plot(iterations, Deltas,  color='b')
# plt.plot(iterations, index, marker='o', linestyle='none', markersize=3, color='b')

plt.show()
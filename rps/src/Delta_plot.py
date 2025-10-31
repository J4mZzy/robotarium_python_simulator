import numpy as np
import matplotlib.pyplot as plt
# Load data
Deltas = np.load('Delta_list.npy')

# Set global font to Times New Roman and enable LaTeX
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

# Plot with different line styles and thickness
iterations = np.arange(Deltas.shape[0])
plt.plot(iterations, Deltas, linestyle='solid', linewidth=3, color='b')

plt.show()
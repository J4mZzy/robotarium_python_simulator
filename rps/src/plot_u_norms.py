import numpy as np
import matplotlib.pyplot as plt

# Load the saved norms
norms = np.load("u_norms.npy")

# Extract the circular and elliptical norms
norm_dxi_cir = norms[:, 0]  # First column
norm_dxi_ellip = norms[:, 1]  # Second column

# Generate time indices (assuming sequential iterations)
iterations = np.arange(len(norm_dxi_cir))

# Set global font to Times New Roman and enable LaTeX
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

# Create figure with larger size
plt.figure(figsize=(8, 5))

# Plot with different line styles and thickness
plt.plot(iterations, norm_dxi_cir, label="Circular Barrier Function $\Vert u\Vert$", 
         linestyle='-', linewidth=5, color='b')

plt.plot(iterations, norm_dxi_ellip, label="Elliptical Barrier Function $\Vert u\Vert$", 
         linestyle='--', linewidth=5, color='r')
# Labels with increased font size
plt.xlabel(r"Time step", fontsize=20)
plt.ylabel(r"$\Vert u\Vert$ $(m/s)$", fontsize=20)

# Improve grid aesthetics
plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

# Improve legend placement and style
plt.legend(fontsize=12, loc="lower right", frameon=True, edgecolor='black')

# Adjust axis ticks for better readability
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Display the plot
plt.show()
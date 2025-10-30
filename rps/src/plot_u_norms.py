import numpy as np
import matplotlib.pyplot as plt

'''This code generates the plot for the robots' u_norm generated from circular CBF, elliptical CBF, and our CBF selection scheme
during the experiment from reading the data saved through the simulations/experiments'''

# Load the saved norms
norms = np.load("u_norms.npy")

# Extract the circular and elliptical norms
norm_dxi_cir = norms[:, 0]  # First column
norm_dxi_ellip = norms[:, 1]  # Second column
norm_varying = norms[:, 2] # Third column
#np.max(np.array([norm_dxi_cir, norm_dxi_ellip]), axis=0)
############ Add other shapes later ########################

# Generate time indices (assuming sequential iterations)
iterations = np.arange(len(norm_dxi_cir))
# print(iterations)

# Set global font to Times New Roman and enable LaTeX
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering

# Create figure with larger size
plt.figure(figsize=(8, 5.5))

# Plot with different line styles and thickness
plt.plot(iterations, norm_dxi_cir, label=r"\textbf{Circular CBF} $\mathbf{\Vert u\Vert}$", 
         linestyle='dashed', linewidth=3, color='b')

plt.plot(iterations, norm_dxi_ellip, label=r"\textbf{Elliptical CBF} $\mathbf{\Vert u\Vert}$", 
         linestyle=':', linewidth=3, color='r')

plt.plot(iterations, norm_varying, label=r"\textbf{Time-Varying CBF} $\mathbf{\Vert u\Vert}$", 
         linestyle='solid', linewidth=4.5, color='purple', alpha=0.5)

# Labels with increased font size
plt.xlabel(r"\textbf{Time step}", fontsize=22)
plt.ylabel(r"$\mathbf{\Vert u\Vert}$ $(m/s)$", fontsize=22)

# Adjust y-axis limits 
plt.ylim(0, max(norm_varying) * 1) 

# plt.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1.02, 1),
#            frameon=True, edgecolor='black', framealpha=1)
plt.legend(fontsize=20,edgecolor = 'black')
plt.legend().set_picker(True)  # Allows clicking to move the legend


# Adjust axis ticks for better readability
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Adjust layout for legend
plt.tight_layout()

# Display the plot
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Load data
trajectories = np.load('trajectories.npy', allow_pickle=True).item()  # dict: i -> (T x 3)
lambs = np.load('lamb_list.npy')       # shape (T, 4) expected
Deltas = np.load('Delta_list.npy')     # shape (T,)
targets = np.load('target_list.npy')   # shape (T,)

# print(Deltas)
# Number of robots
N = 8

# Time steps (assumes key 0 exists)
T = np.array(trajectories[0]).shape[0]
h_min = np.full(T, np.inf)

# ---- Barrier parameters (once) ----
r = 0.25
a = 0.25*0.95 
b = 0.25 
p = 3
safety_width = 0.4
rt3 = np.sqrt(3.0)  # for the triangular barrier

# (optional) make trajectories arrays once
traj = {i: np.asarray(trajectories[i]) for i in range(N)}  # each is (T, 3)

def logsumexp3(L1, L2, L3):
    """Stable log-sum-exp for 3 scalars."""
    m = max(L1, L2, L3)
    return m + np.log(np.exp(L1 - m) + np.exp(L2 - m) + np.exp(L3 - m))

for t in range(T):
    # loop unique unordered pairs (i<j). If you want all ordered pairs, use permutations(range(N),2)
    for i in range(N - 1):
        xi = traj[i][t, :]          # (x, y, theta)
        for j in range(i + 1, N):
            xj = traj[j][t, :]

            # relative pose in world frame
            ex = xi[0] - xj[0]
            ey = xi[1] - xj[1]

            # ---- Circle CBF ----
            h_ij_cir = (ex * ex /r**2 + ey * ey/r**2) - 1

            # ---- Rotate into agent i frame (do NOT overwrite ex/ey) ----
            c = np.cos(xi[2]); s = np.sin(xi[2])
            u =  c * ex + s * ey                 # x in i-frame
            v = s * ex - c * ey                 # y in i-frame

            # ---- Ellipse CBF ----
            h_ij_ell = (u / a) ** 2 + (v / b) ** 2 - 1.0

            # ---- L^p "square" CBF ----
            h_ij_square = (np.abs(u) ** p + np.abs(v) ** p) ** (1.0 / p) - safety_width / 2.0

            # ---- Log-sum-exp "triangle" CBF (stable) ----
            L1 = 4 * u + 4 * rt3 * v
            L2 = -8 * u
            L3 = 4 * u - 4 * rt3 * v
            h_ij_tri = (3.0 / 5.0) * logsumexp3(L1, L2, L3) - 1.0

            # previous blended barrier with lambdas at time t
            lam = lambs[t]  # expect length 4
            h_prev = lam[0] * h_ij_cir + lam[1] * h_ij_ell + lam[2] * h_ij_tri + lam[3] * h_ij_square

            # morph toward target
            Delta_t = Deltas[t]
            tgt = targets[t]
            # print(tgt)
            if tgt == 1:
                h_ij = (1 - Delta_t) * h_prev + Delta_t * h_ij_cir
                # h_ij = h_ij_cir
            elif tgt == 2:
                h_ij = (1 - Delta_t) * h_prev + Delta_t * h_ij_ell
                # h_ij = h_ij_ell
            elif tgt == 3:
                h_ij = (1 - Delta_t) * h_prev + Delta_t * h_ij_tri
            elif tgt == 4:
                h_ij = (1 - Delta_t) * h_prev + Delta_t * h_ij_square
            else:
                h_ij = h_prev  # fallback if target out of range
            # h_ij = h_ij_ell

            # ---- take the minimum across pairs (INSIDE the loops) ----
            if h_ij < h_min[t]:
                h_min[t] = h_ij

# Vars for plot
iterations = np.arange(T)
h_min_global = float(np.min(h_min))
index = int(np.argmin(h_min))
zeros = np.zeros(T)   # <-- was np.empty(T)

print(h_min_global)

# Plot
plt.figure(figsize=(8, 5.5))
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True

plt.plot(iterations, h_min, linestyle='solid', linewidth=3, color='b')
plt.plot(iterations, zeros, linestyle='dashed', linewidth=3, color='r')

plt.xlabel(r"\textbf{Time step}", fontsize=22)
plt.ylabel(r"$\mathbf{h_{\mathbf{\min}}}$", fontsize=22)
plt.xticks(fontsize=18); plt.yticks(fontsize=18)
# plt.savefig("h_min_20_agents", dpi=600)
plt.show()

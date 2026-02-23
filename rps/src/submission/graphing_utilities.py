#
# Utilities for graphing in experiments
#

from matplotlib.collections import LineCollection
import numpy as np
from matplotlib import contour

def lse3(L1, L2, L3):
    """Stable log-sum-exp for arrays."""
    M = np.maximum.reduce([L1, L2, L3])
    return M + np.log(np.exp(L1 - M) + np.exp(L2 - M) + np.exp(L3 - M))

def densify_segments(segments, max_step):
    """Insert points so edges are <= max_step; segs: list of (m x 2) arrays."""
    out = []
    for seg in segments:
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

def init_hvis(ax, N, CM, *, radius, a, b, w, Rvis_scale=1.4, grid_res=201, line_w=2):
    """
    ax: matplotlib axes (e.g., r.axes), N: #agents, CM: (N x 3/4) colors
    returns a dict 'H' with precomputed fields and artists.
    """
    radius = float(radius); a = float(a); b = float(b); w = float(w)

    # local grid in (u,v) — agent frame
    Rvis = Rvis_scale * max(radius, a, b, w/2.0)
    nu = nv = int(grid_res)
    u_vals = np.linspace(-Rvis, Rvis, nu)
    v_vals = np.linspace(-Rvis, Rvis, nv)
    UU, VV = np.meshgrid(u_vals, v_vals)

    # fields that only change if radius/a/b/w/p change
    h_circ_l  = (UU/radius)**2 + (VV/radius)**2 - 1.0
    h_ellip_l = (UU/a)**2      + (VV/b)**2      - 1.0
    h_ellip_r = (UU/b) ** 2 + (VV/a) ** 2 - 1.0
    h_sq_l = np.abs(UU/w)**3 + np.abs(VV/w)**3 - 1.0
    h_fields_local = [h_circ_l, h_ellip_l, h_ellip_r, h_sq_l]

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
        c  = 1; s = 0

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
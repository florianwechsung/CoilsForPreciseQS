import numpy as np
import matplotlib
font = {'family' : 'serif',
        'size'   : 14}

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt


x = []
y = []
z = []
idx = []
# for j in range(3):
for j in [2]:
    phi = (j/4)*(2*np.pi/2)

    M = np.loadtxt(f'poincare/output_well_False_lengthbound_24_0_kap_5_0_msc_5_0_dist_0_1_fil_0_ig_2_order_16_expquad__poincare_{j}.txt', delimiter=',', skiprows=1)
    # M = np.loadtxt(f'poincare/output_well_True_lengthbound_24_0_kap_5_0_msc_5_0_dist_0_1_fil_0_ig_5_order_16__poincare_{j}.txt', delimiter=',', skiprows=1)

    n = M.shape[1]//2-1
    n = 28

    plt.figure()
    for i in range(n):
        r = M[:, 2*i]
        x.extend(list(np.cos(phi) * r))
        y.extend(list(np.sin(phi) * r))
        z.extend(list(M[:, 2*i+1]))
        idx.extend([i] * M.shape[0])
        plt.scatter(r, list(M[:, 2*i+1]), s=1.0, c="r", marker='o', linewidths=0)

    plt.plot(M[:, -2], M[:, -1], c="b", linewidth=3)

    ax = plt.gca()
    ax.axis('equal')
    ax.set_xlabel("$r$")
    ax.set_ylabel("$z$")
    plt.savefig(f"/tmp/{j}.png", dpi=400, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

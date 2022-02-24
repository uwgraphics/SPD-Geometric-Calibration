"""
Generate a box plot of average errors as the amount of noise varies
"""
from calibration.util import *
from calibration.solver import bf_slsqp, slsqp

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# change working directory to the directory this file is in (for saving data)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

SENSOR_NOISES = [0, 5, 10, 15]
NUM_OBS = 32
SAMPLES = 100
GEN_DATA = False

if(GEN_DATA):
    results = {
        "noise": [],
        "p_error": [],
        "u_error": [],
        "a_error": [],
        "d_error": []
    }
    # will average over later
    avg_results = pd.DataFrame(results.copy())

    for noise in SENSOR_NOISES:

        # use a fixed plane and true p, u soln for simplicity
        a = np.array([0, 0, -1])
        d = -500
        p = np.array([0, 0, 0])
        u = np.array([0, 0, -1])

        center = np.array([0, 0, -500])

        for i in range(SAMPLES):
            print(i)

            tfs = generate_motions(p, u, a, d, center, [[-1000, 1000]]*3, radius=2000, n=NUM_OBS)

            tfd_ps = [from_hom(tf @ to_hom(p)) for tf in tfs]
            tfd_us = [from_hom(tf @ np.append(u, [0])) for tf in tfs]

            measurements = [gen_observation(tfd_p, tfd_u, a, d)[0] for tfd_p, tfd_u in zip(tfd_ps, tfd_us)]
            measurements = [m + np.random.normal(0, noise) for m in measurements]

            soln, loss = bf_slsqp(
                tfs,
                measurements,
                p_bounds=[
                    [-100, 100],
                    [-100, 100],
                    [-100, 100]
                ],
                d_bounds = [-500, 500]
            )

            results["noise"].append(noise)
            results["p_error"].append(np.linalg.norm(np.array(p) - np.array(soln[0]))) #TODO check
            results["u_error"].append(np.degrees(angle_between(u, soln[1])))
            results["a_error"].append(np.degrees(angle_between(a, soln[2])))
            results["d_error"].append(np.abs(d - soln[3]))

    results = pd.DataFrame(results)

    results.to_csv('data/simulated/vary_noise.csv')

else:
    results = pd.read_csv("data/simulated/vary_noise.csv")

fig, ax = plt.subplots(1, 2)

sns.boxplot(ax = ax[0], x="noise", y="p_error", data=results, color="lightgray")
ax[0].set_xlabel("                                                   \u03c3 of Gaussian sensor noise (mm)", fontsize=12)
ax[0].set_ylabel("Error in solved position (mm)", fontsize=12)

sns.boxplot(ax = ax[1], x="noise", y="u_error", data=results, color="lightgray")
ax[1].set_xlabel("")
ax[1].set_ylabel("Error in solved orientation (degrees)", fontsize=12)
fig.set_size_inches(6, 4)

plt.tight_layout()
plt.savefig("figures/vary_noise.png", dpi=200)
plt.show()
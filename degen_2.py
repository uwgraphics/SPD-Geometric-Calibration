from calibration.util import *
from calibration.solver import *

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# change working directory to the directory this file is in (for saving data)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

NUM_OBS = 32
MEASURE_NOISE = 0.5
DIST_NOISES = np.linspace(1, 11, 6)
NUM_SAMPLES = 50
GEN_DATA = False

if(GEN_DATA):
    results = {
        "dist_noise": [],
        "p_error": []
    }

    # use a fixed plane and true p, u soln for simplicity
    a = np.array([0, 0, 1])
    d = 500
    start_p = np.array([0, 0, 0])
    start_u = np.array([0, 0, -1])

    for dist_noise in DIST_NOISES:
        print("dist_noise", dist_noise)
        errors = []
        for _ in range(NUM_SAMPLES):
            plane_pts = scattered_on_plane(a, d, [0, 0, -500], 100, NUM_OBS)

            # generate distances and scale them so avg. difference from 100 is dist_noise
            distances = np.random.uniform(0, 1, NUM_OBS)
            distances = (distances * (dist_noise / np.average(distances))) + 100

            ps = []
            us = []
            for i in range(NUM_OBS):
                p = np.random.uniform(-300, 300, 3)
                u = np.array(plane_pts[i]) - p #TODO this could be backwards
                u = u / np.linalg.norm(u)

                ps.append(plane_pts[i] - (u*distances[i]))
                us.append(u)

            tfs = points_to_transforms([start_p, *ps], [start_u, *us])

            measurements = [gen_observation(p, u, a, d)[0] for p, u in zip(ps, us)]

            for m, u, p in zip(measurements, us, ps):
                # check that the point projected out from p and u is on plane
                x = (np.array(u) * m) + np.array(p)
                res = np.dot(np.array(a), x) + d
                assert(np.abs(res) < 0.01)

            # add noise to measurements
            measurements = [m + np.random.normal(0, MEASURE_NOISE) for m in measurements]

            soln, _ = bf_slsqp(
                tfs,
                measurements,
                [[-10000, 10000]]*3,
                [-10000, 10000]
            )

            p_error = np.linalg.norm(np.array(start_p) - np.array(soln[0]))

            results["dist_noise"].append(dist_noise)
            results["p_error"].append(p_error)

    results = pd.DataFrame(results)
    results.to_csv("data/simulated/degen_2.csv")

else:
    results = pd.read_csv("data/simulated/degen_2.csv")

plt.figure(figsize=(6,4))
ax = sns.boxplot(x="dist_noise", y="p_error", data=results, color="lightgray")
ax.set_xlabel("\u03c3 of distance measurements (mm)", fontsize=14)
ax.set_ylabel("Error in solved position (mm)", fontsize=14)

plt.subplots_adjust(
    top=0.907,
    bottom=0.146,
    left=0.118,
    right=0.95,
    hspace=0.2,
    wspace=0.2
)
plt.tight_layout()
plt.savefig("figures/degen_2.png", dpi=200)
plt.show()
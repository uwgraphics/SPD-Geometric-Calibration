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
RADII = [0.1, 0.4, 0.7, 1, 1.3, 1.6, 1.9]
NUM_SAMPLES = 100
GEN_DATA = False

if(GEN_DATA):
    results = {
        "radius": [],
        "p_error": [],
        "u_error": [],
        "a_error": []
    }

    # use a fixed plane and true p, u soln for simplicity
    a = np.array([0, 0, 1])
    d = 500
    start_p = np.array([0, 0, 0])
    start_u = np.array([0, 0, -1])

    for radius in RADII:
        print("radius", radius)
        errors = []
        for _ in range(NUM_SAMPLES):
            bbox = [
                [-400, 400],
                [-400, 400],
                [-400, 400]
            ]
            tfs = generate_motions(start_p, start_u, a, d, np.array([0, 0, -500]), bbox, radius=radius, n=NUM_OBS)

            ps = [from_hom(tf @ to_hom(np.array(start_p))) for tf in tfs]
            us = [from_hom(tf @ np.append(start_u, [0])) for tf in tfs]

            measurements = [gen_observation(p, u, a, d)[0] for p, u in zip(ps, us)]

            for m, u, p in zip(measurements, us, ps):
                # check that the point projected out from p and u is on plane
                x = (np.array(u) * m) + np.array(p)
                res = np.dot(np.array(a), x) + d
                assert(np.abs(res) < 0.01)

            # add noise to measurements
            measurements = [m + np.random.normal(0, MEASURE_NOISE) for m in measurements]

            soln, _ = slsqp(
                tfs,
                measurements,
                a,
                d,
                start_p,
                start_u,
                [[-10000, 10000]]*3,
                [-10000, 10000]
            )

            p_error = np.linalg.norm(np.array(start_p) - np.array(soln[0]))
            u_error = np.degrees(angle_between(start_u, soln[1]))
            a_error = np.degrees(angle_between(a, soln[2]))

            print(a_error)

            results["radius"].append(radius)
            results["p_error"].append(p_error)
            results["u_error"].append(u_error)
            results["a_error"].append(a_error)
        
    results = pd.DataFrame(results)
    results.to_csv("data/simulated/degen_3.csv")

else:
    results = pd.read_csv("data/simulated/degen_3.csv")

plt.figure(figsize=(6,4))
ax = sns.boxplot(x="radius", y="a_error", data=results, color="lightgray")
ax.set_xlabel("Radius of observed point spread on plane (mm)", fontsize=14)
ax.set_ylabel("Error in solved a vector (degrees)", fontsize=14)

plt.tight_layout()
plt.savefig("figures/degen_3.png", dpi=200)
plt.show()
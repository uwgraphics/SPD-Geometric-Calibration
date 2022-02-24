from calibration.util import *
from calibration.solver import *

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.transform import Rotation as R

# change working directory to the directory this file is in (for saving data)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

NUM_OBS = 32
MEASURE_NOISE = 0.5
U_NOISES = np.linspace(5, 30, 6)
NUM_SAMPLES = 50
GEN_DATA = False

if(GEN_DATA):
    results = {
        "u_noise": [],
        "p_error": []
    }
    for u_noise in U_NOISES:
        print("u_noise", u_noise)
        for _ in range(NUM_SAMPLES):
            # generate a unit vector - this will remain fixed in the degenerate case
            u = []
            us = []
            u = [np.random.uniform(-1, 1) for _ in range(3)]
            u = u / np.linalg.norm(u)
            us = [u]*NUM_OBS

            # generate random angles to apply to the unit vectors, then
            # scale so that the average rotation applied is u_noise
            angles = np.random.uniform(0, 1, NUM_OBS)
            angles = angles * (u_noise / np.average(angles))

            # add some noise to unit vectors
            random_us = []
            for i in range(NUM_OBS):
                # generate a random vector and make it orthogonal to x
                # this will be the axis for our axis angle rotation
                # https://stackoverflow.com/questions/33658620/generating-two-orthogonal-vectors-that-are-orthogonal-to-a-particular-direction
                axis = np.random.randn(3)
                axis -= axis.dot(u) * u / np.linalg.norm(u)**2 
                axis /= np.linalg.norm(axis)

                rot = R.from_rotvec(np.radians(angles[i]) * axis)

                new_u = rot.apply(u)

                random_us.append(new_u)
            
            us = random_us

            a = -u
            d = np.random.uniform(400, 100)

            ps = []
            measurements = []
            i = 0
            while i < NUM_OBS:
                # generate a random cloud of points
                p = [np.random.uniform(-100, 100) for _ in range(3)]

                obs = gen_observation(p, us[i], a, d)

                if(obs != float('inf')):
                    measurements.append(obs[0])
                    ps.append(p)

                    # check that the point projected out from p and u is on plane
                    x = (np.array(us[i]) * obs[0]) + np.array(p)
                    res = np.dot(np.array(a), x) + d
                    assert(np.abs(res) < 0.01)

                    i += 1

            tfs = [ID, *points_to_transforms(ps, us)]

            # add noise to measurements
            measurements = [m + np.random.normal(0, MEASURE_NOISE) for m in measurements]

            soln, _ = bf_slsqp(
                tfs,
                measurements,
                [[-10000, 10000]]*3,
                [-10000, 10000]
            )

            p_error = np.linalg.norm(np.array(ps[0]) - np.array(soln[0]))

            results["u_noise"].append(u_noise)
            results["p_error"].append(p_error)
        

    results = pd.DataFrame(results)
    results.to_csv('data/simulated/degen_1.csv')

else:
    results = pd.read_csv("data/simulated/degen_1.csv")

plt.figure(figsize=(6,4))
ax = sns.boxplot(x="u_noise", y="p_error", data=results, color="lightgray")
ax.set_xlabel("Average angle between two observations (degrees)", fontsize=14)
ax.set_ylabel("Error in solved position (mm)", fontsize=14)

plt.tight_layout()
plt.savefig("figures/degen_1.png", dpi=200)
plt.show()
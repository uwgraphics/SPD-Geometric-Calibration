"""
Comparison between solving with only one initial estimate and taking the best of
6 initial estimates (corresponding to the 6 axis-aligned unit vectors) for u.
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

SENSOR_NOISES = [0, 20, 40]
SAMPLES = 100
GEN_DATA = False

if(GEN_DATA):
    results = {
        "solver_type": [],
        "noise": [],
        "p_error": [],
        "u_error": [],
        "a_error": [],
        "d_error": []
    }

    for noise in SENSOR_NOISES:
        i = 0
        while i < SAMPLES:
            p = [np.random.uniform(-100, 100) for _ in range(3)]

            u = random_unit_vector()

            a = random_unit_vector()

            d = np.random.uniform(-200, 200)

            obs = gen_observation(p, u, a, d)

            if(obs != float('inf') and angle_between(u, a) < 1):
                print(i)
                x_0 = obs[1]

                tfs = generate_motions(p, u, a, d, x_0, [[-1000, 1000]]*3, radius=2000, n=32)

                tfd_ps = [from_hom(tf @ to_hom(p)) for tf in tfs]
                tfd_us = [from_hom(tf @ np.append(u, [0])) for tf in tfs]

                measurements = [gen_observation(tfd_p, tfd_u, a, d)[0] for tfd_p, tfd_u in zip(tfd_ps, tfd_us)]
                measurements = [m + np.random.normal(0, noise) for m in measurements]

                soln, loss = slsqp(
                    tfs,
                    measurements,
                    a_est=[0, 0, 1],
                    d_est=0,
                    p_est=[0, 0, 0],
                    u_est=[0, 0, -1],
                    p_bounds=[
                        [-100, 100],
                        [-100, 100],
                        [-100, 100]
                    ],
                    d_bounds = [-200, 200]
                )

                results["solver_type"].append("One")
                results["noise"].append(noise)
                results["p_error"].append(np.linalg.norm(np.array(p) - np.array(soln[0]))) #TODO check
                results["u_error"].append(angle_between(u, soln[1]))
                results["a_error"].append(angle_between(a, soln[2]))
                results["d_error"].append(np.abs(d - soln[3]))

                soln, loss = bf_slsqp(
                    tfs,
                    measurements,
                    p_bounds=[
                        [-100, 100],
                        [-100, 100],
                        [-100, 100]
                    ],
                    d_bounds = [-200, 200]
                )

                results["solver_type"].append("Best of 6")
                results["noise"].append(noise)
                results["p_error"].append(np.linalg.norm(np.array(p) - np.array(soln[0]))) #TODO check
                results["u_error"].append(angle_between(u, soln[1]))
                results["a_error"].append(angle_between(a, soln[2]))
                results["d_error"].append(np.abs(d - soln[3]))

                i+=1

    results = pd.DataFrame(results)

    results.to_csv('data/simulated/initial_est_test.csv')

else:
    results = pd.read_csv("data/simulated/initial_est_test.csv")

plt.figure(figsize=(6,4))
ax = sns.stripplot(x="noise", y="p_error", hue="solver_type", data=results, dodge=True)

ax.set_xlabel("\u03c3 of gaussian sensor noise (mm)", fontsize=12)
ax.set_ylabel("Error in recovered sensor \n position (mm)", fontsize=12)
plt.legend(title='', labels=["One initial estimate", "Best of 6 estimates"], loc='center', bbox_to_anchor=(0.5, -0.25), ncol=2)

plt.tight_layout()
plt.savefig("figures/initial_est_test.png", dpi=200)
plt.show()
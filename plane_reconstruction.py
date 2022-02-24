from calibration.util import *
from calibration.solver import *

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

np.set_printoptions(suppress=True)

# change working directory to the directory this file is in (for saving data)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = "data/real-world"
GEN_DATA = False
PERTURBED_N = 25

if(GEN_DATA):
    results = {
        "calibrate_trial": [],
        "perturbed": [],
        "avg_residual": [],
        "sensor": [],
        "col_trial": [] # for plotting with seaborn's catplot
    }

    for pose_code in ["P1", "P2", "P3", "P4"]:
        filtered_dirs = [d for d in os.listdir(DATA_DIR) if d[-2:] == pose_code]
        for calibrate_dir in filtered_dirs:
            test_dirs = [d for d in filtered_dirs if d != calibrate_dir]

            # make directories full paths again, so they can be passed to util fns
            calibrate_dir = os.path.dirname(os.path.abspath(__file__)) + "/" + DATA_DIR + "/" + calibrate_dir
            test_dirs = [os.path.dirname(os.path.abspath(__file__)) + "/" + DATA_DIR + "/" + d for d in test_dirs]
            
            # calibrate using the data in calibrate_dir
            measurements, transforms = read_data(calibrate_dir)
            soln, _ = bf_slsqp(
                transforms,
                measurements,
                [[-1000, 1000]]*3,
                [-1000, 1000]
            )
            calib_p = np.array(soln[0])
            calib_u = np.array(soln[1])

            perturbed_ps = []
            perturbed_us = []
            for _ in range(PERTURBED_N):
                perturbed_ps += perturb_p(calib_p, 10, True)
                perturbed_us += perturb_u(calib_u, 10, True)
            
            for i, new_calib_p, new_calib_u in zip(range(len(perturbed_ps)+1), [calib_p] + perturbed_ps, [calib_u] + perturbed_us):
                # test on each trial for which the sensor was in the same pose
                avg_residuals = []
                for test_dir in test_dirs:
                    measurements, transforms = read_data(test_dir)

                    # apply each transform in test trial to the solved sensor pose from
                    # calibration trial
                    tfd_ps = [from_hom(tf @ to_hom(new_calib_p)) for tf in transforms]
                    tfd_us = [from_hom(tf @ np.append(new_calib_u, 0)) for tf in transforms]

                    # project out measurements from transformed poses (p, u) to get
                    # points that lie on the calibration plane
                    pts = [p + m*u for p, m, u in zip(tfd_ps, tfd_us, measurements)]

                    # fit a plane to those points and measure the sum of residuals
                    a, d, res = fit_plane(pts)
                    avg_residuals.append(res / len(measurements))

                results["calibrate_trial"].append(calibrate_dir.split("/")[-1])
                results["avg_residual"].append(np.average(avg_residuals))
                if(i == 0):
                    results["perturbed"].append("Solution")
                else:
                    results["perturbed"].append("Perturbed")

                if(pose_code in ["P1", "P2"]):
                    results["sensor"].append("VL53L3CX")
                else:
                    results["sensor"].append("VL6180X")

                fold = calibrate_dir.split("/")[-1]
                if(fold.split("_")[-1] in ["P1", "P2"]):
                    results["col_trial"].append(fold[-5:])
                else:
                    if(fold[-1] == "3"):
                        results["col_trial"].append((fold[:-1] + "1")[-5:])
                    else:
                        results["col_trial"].append((fold[:-1] + "2")[-5:])

    results = pd.DataFrame(results)
    results.to_csv('data/evaluation/plane_reconstruction.csv')

else:
    results = pd.read_csv('data/evaluation/plane_reconstruction.csv')

g = sns.catplot(
    x="col_trial",
    y="avg_residual",
    hue="perturbed",
    col="sensor",
    data=results,
    kind="strip",
    height=4,
    aspect=0.66,
    legend_out=True,
    legend=False
)
(g.set_axis_labels("Trial Number", "Average Residual to Plane (mm)", fontsize=12)
  .set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])
  .set_titles("{col_name}", size=12)
  .set(ylim=(0, 16))
)
g.axes[0][0].set_xticklabels([9, 10, 11, 12, 13, 14, 15, 16])

plt.tight_layout()
plt.savefig("figures/plane_reconstruction.png", dpi=200)
plt.show()
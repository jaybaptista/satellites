import dgf
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord

files = ["draco", "draco_a", "draco_b", "draco_c"]

ts = np.linspace(0.1, 0.95, num=20)

for i, gal in enumerate(files):
    prob = dgf.prob_utils.Probability(gal)
    obs = prob.obs

    cm_widths = [0.1, 0.2, 0.5]

    ps = np.zeros([len(ts), len(cm_widths)])
    cs = np.zeros([len(ts), len(cm_widths)])

    for i, thresh in enumerate(ts):
        for j, width in enumerate(cm_widths):
            p_i = prob.cm(sigma=width)
            m = ((p_i) / np.max(p_i)) > thresh
            tot = sum(m)

            c = sum(m & (obs["member"] == 1)) / sum((obs["member"] == 1))
            p = sum(m & (obs["member"] == 1)) / tot

            ps[i, j] = p
            cs[i, j] = c

    fig, ax = plt.subplots(1, 2, dpi=200, figsize=(10, 5))
    for j in np.arange(0, len(cm_widths)):
        ax[0].plot(ts, ps[:, j], label=rf"$\sigma = {cm_widths[j]}$")
        ax[1].plot(ts, cs[:, j])
    ax[0].legend()

    ax[0].set_title("Purity")
    ax[1].set_title("Completeness")

    # ax[0].set_yscale("log")
    # ax[1].set_yscale("log")
    plt.savefig(f"plots/pc_cm_{gal}.png")

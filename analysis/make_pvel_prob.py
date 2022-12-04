import dgf
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord

files = ["draco", "draco_a", "draco_b", "draco_c"]
fg_data = pd.read_csv("./bg_characterize/foregrounds.csv")

ts = np.linspace(0.1, 0.95, num=20)

data = {
    "galaxy": [],
    "dual.metal.cont": [],
    "dual.cont": [],
    "b": [],
    "l": [],
}

for i, gal in enumerate(files):
    prob = dgf.prob_utils.Probability(gal)

    mw_theta = [0.5, fg_data["hrv"][1], fg_data["dhrv"][1], 0, 0.5]

    adv_prob = prob.dispersion_adv(mw_theta=[0.5, -10, 50, 0, 0.5])
    reg_prob = prob.dispersion(niter=5000)

    obs = prob.obs
    m = sum((adv_prob / np.max(adv_prob) > 0.1) & (obs["member"] == 0))
    m2 = sum((reg_prob / np.max(reg_prob) > 0.1) & (obs["member"] == 0))

    ps_adv = []
    cs_adv = []

    ps_reg = []
    cs_reg = []

    for thresh in ts:
        m_adv = ((adv_prob) / np.max(adv_prob)) > thresh
        m_reg = ((reg_prob) / np.max(reg_prob)) > thresh

        tot_adv = sum(m_adv)
        tot_reg = sum(m_reg)

        c_adv = sum(m_adv & (obs["member"] == 1)) / sum((obs["member"] == 1))
        c_reg = sum(m_reg & (obs["member"] == 1)) / sum((obs["member"] == 1))

        p_adv = sum(m_adv & (obs["member"] == 1)) / tot_adv
        p_reg = sum(m_reg & (obs["member"] == 1)) / tot_reg

        ps_adv.append(p_adv)
        ps_reg.append(p_reg)

        cs_adv.append(c_adv)
        cs_reg.append(c_reg)

    fig, ax = plt.subplots(1, 2, dpi=200, sharey=True, figsize=(5, 2.5))

    ax[0].plot(ts, ps_adv, c="b", label="adv")
    ax[1].plot(ts, cs_adv, c="b")
    ax[0].plot(ts, ps_reg, c="r", label="reg")
    ax[1].plot(ts, cs_reg, c="r")

    ax[0].legend()

    ax[0].set_title("Purity")
    ax[1].set_title("Completeness")

    plt.savefig(f"plots/pc_pvel_{gal}.png")

    coords = SkyCoord(prob.ra_0, prob.dec_0)
    lat = coords.galactic.b
    long = coords.galactic.l

    data["galaxy"].append(files[i])
    data["dual.metal.cont"].append(m)
    data["dual.cont"].append(m2)
    data["b"].append(lat)
    data["l"].append(long)

df = pd.DataFrame(data)
df.to_csv("probs_pvel.csv")

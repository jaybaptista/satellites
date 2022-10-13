import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython import embed


def view(gal_name):

    obs = pd.read_hdf(f"./gals/{gal_name}_observation.hdf")
    stars = pd.read_hdf(f"./gals/{gal_name}_stars.hdf")
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=200)

    ax = axs.flatten()

    m_mask = obs["member"] == 1

    ### RA DEC plot ###
    ax[0].scatter(obs["RA"][m_mask], obs["DEC"][m_mask], c="b", s=1.5, alpha=0.2)

    ax[0].scatter(
        obs["RA"][~m_mask], obs["DEC"][~m_mask], fc="None", ec="k", s=5, marker="o"
    )

    ax[0].set_aspect(1)
    ax[0].set_xlabel("RA")
    ax[0].set_ylabel("Dec")

    ### CMD ###

    ax[1].invert_yaxis()
    ax[1].scatter(
        obs["g"][~m_mask] - obs["r"][~m_mask],
        obs["r"][~m_mask],
        fc="None",
        ec="k",
        s=5,
        marker="o",
        label="MW",
    )
    ax[1].scatter(
        obs["g"][m_mask] - obs["r"][m_mask],
        obs["r"][m_mask],
        c="b",
        s=1.5,
        alpha=0.5,
        label="Member",
    )
    ax[1].set_ylim(23, 17)
    ax[1].legend()
    ax[1].set_xlabel("$g-r$")
    ax[1].set_ylabel("$r$")

    ### HRV ###
    _, bins, _ = ax[2].hist(obs["HRV"], bins=30, histtype="step", color="k")
    ax[2].hist(obs["HRV"][m_mask], bins=bins, color="b")
    ax[2].set_xlabel("HRV [km/s]")

    ### FeH ###
    _, bins, _ = ax[3].hist(obs["[Fe/H]"], bins=30, histtype="step", color="k")
    ax[3].hist(obs["[Fe/H]"][m_mask], bins=bins, color="b")
    ax[3].set_xlabel("[Fe/H]")

    plt.savefig(f"./plots/{gal_name}_observation.png", bbox_inches="tight")


view("draco_a")
view("draco_b")
view("draco_c")

view("kona_a")
view("kona_b")
view("kona_c")

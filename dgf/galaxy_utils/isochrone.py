import matplotlib.pyplot as plt
import astropy.io.ascii as asc
import numpy as np
import pandas as pd
import astropy.units as u


def formatBranches(isochrone, mag_lim=6):
    m = (isochrone["label"] < 4) & (isochrone["rmag"] < mag_lim)
    return isochrone[m]


def interpolateIsochrone(isochrone):
    polynomials = {"m": [], "p1": [], "p2": []}

    for i in np.arange(len(isochrone) - 1):
        dy = isochrone["rmag"][i] - isochrone["rmag"][i + 1]
        dx = (isochrone["gmag"] - isochrone["rmag"])[i] - (
            isochrone["gmag"] - isochrone["rmag"]
        )[i + 1]

        if dx != 0:
            m = dy / dx
            polynomials["m"].append(m)
            polynomials["p1"].append(
                [isochrone["rmag"][i], (isochrone["gmag"] - isochrone["rmag"])[i]]
            )
            polynomials["p2"].append(
                [
                    isochrone["rmag"][i + 1],
                    (isochrone["gmag"] - isochrone["rmag"])[i + 1],
                ]
            )

    return polynomials


def mapIso(template, lumfunc):
    iso_pts = []
    lines = interpolateIsochrone(template)

    bin_df = lumfunc["rmag"] / np.sum(lumfunc["rmag"])

    for i, bin in enumerate(lumfunc["magbinc"]):
        idx = np.where(
            (bin < np.array(lines["p1"])[:, 0]) & (bin > np.array(lines["p2"])[:, 0])
        )[0]

        if len(idx) > 0:
            m = lines["m"][idx[0]]
            p1 = lines["p1"][idx[0]]
            p2 = lines["p2"][idx[0]]
            iso_pts.append([bin, (((bin - p1[0]) / m) + p1[1]), bin_df[i]])

    iso_pts = np.array(iso_pts)
    iso_pts[:, 2] = iso_pts[:, 2] / np.sum(iso_pts[:, 2])
    return iso_pts


def sampleIsochrone(r, color, prob, N=1000, noise=0.2):
    if len(r) != len(color):
        raise ValueError(
            "Length of color array not the same as length of magnitude array"
        )

    sample = {"mag": [], "color": []}

    n = len(r)
    rand_idx = np.random.choice(np.arange(0, n), p=prob, size=N, replace=True)

    for i in rand_idx:
        sample["mag"].append(r[i] + np.random.normal(0, noise))
        sample["color"].append(color[i] + np.random.normal(0, noise))

    return sample


def sample(isochrone, lumfunc, N, noise=0.2, dist=0 * u.kpc):
    isochrone = formatBranches(isochrone)
    polynomials = interpolateIsochrone(isochrone)
    mock_iso = mapIso(isochrone, lumfunc)

    rmag = None

    if dist.value == 0:
        rmag = mock_iso[:, 0]
    else:
        rmag = 5 * np.log10(dist.to(u.pc).value) - 5 + mock_iso[:, 0]

    color = mock_iso[:, 1]
    prob = mock_iso[:, 2]

    return sampleIsochrone(rmag, color, prob, N, noise)
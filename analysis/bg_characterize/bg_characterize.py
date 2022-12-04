import numpy as np
import astropy.io.ascii as asc
from scipy.stats import norm
from dgf import GaussianFit
from os import path
import pandas as pd

dir = "../../foregrounds/"

foregrounds = [
    "ra_260.0_dec_57.9_sa_20deg.txt",
    "ra_309_dec_48.6_sa_20deg.txt",
    "ra_335_dec_26.8_sa_20deg.txt",
    "ra_353_dec_1.9_sa_20deg.txt",
]

star_count = 200

niter = 2000
nwalkers = 20

data = {"file": foregrounds, "hrv": [], "dhrv": []}

for i, path in enumerate(foregrounds):

    fg_path = dir + path

    fg_data = asc.read(fg_path, format="commented_header")
    mw_hrv = fg_data["HRV"]
    mw_dhrv = fg_data["errHrv"]
    sample_indices = np.random.choice(
        np.arange(0, len(mw_hrv)), star_count, replace=False
    )

    fitter = GaussianFit(
        x_lims=(-300, 300), s_lims=(0.5, 500), niter=niter, walkers=nwalkers,
    )

    _, _, _, sampler = fitter.run(
        mw_hrv[sample_indices], mw_dhrv[sample_indices], -90, 20
    )

    samples = sampler.flatchain

    res = fitter.get_results(samples)

    fit = {"HRV": res[0], "HRV_err": res[1], "dHRV": res[2], "dHRV_err": res[3]}

    data["hrv"].append(fit["HRV"])
    data["dhrv"].append(fit["dHRV"])

    fitter.plot(
        samples, labels=["v", r"$\sigma_v$"], save_to=f"{path}_vel.png",
    )

df = pd.DataFrame(data)
df.to_csv("./foregrounds.csv")

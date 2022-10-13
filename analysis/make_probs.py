import argparse
import dgf
from IPython import embed
import numpy as np
import astropy.units as u

parser = argparse.ArgumentParser(description="Calculates probabilities.")
parser.add_argument(
    "galaxy", type=str, help="Name of galaxy, matched to local galaxy catalog."
)
parser.add_argument(
    "frac",
    nargs="?",
    const=1,
    type=float,
    help="Cutoff fraction, assumes probabilities calculated beforehand.",
    default=0.95,
)
parser.add_argument(
    "reff",
    nargs="?",
    const=1,
    type=float,
    help="Cutoff fraction, assumes probabilities calculated beforehand.",
    default=8 * u.arcmin,
)

args = parser.parse_args()

prob = dgf.prob_utils.Probability(args.galaxy)

cprobs = prob.cm(sigma=0.1, save_plot=True)
rprobs = prob.radial(reff=args.reff, save_plot=True)
vprobs = prob.dispersion(save_plot=True)


obs_file = prob.obs
obs_file["Prob.cm"] = cprobs
obs_file["Prob.reff"] = rprobs
obs_file["Prob.vel"] = vprobs

net_prob = obs_file["Prob.cm"] * obs_file["Prob.reff"]

obs_file["Prob"] = net_prob

obs_file.to_hdf(f"./data/prob_df_{args.galaxy}.hdf", key="w")
prob.recovery_fn(cutoff=args.frac, save_plot=True)

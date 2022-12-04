import numpy as np
import astropy.units as u
import json
import astropy.io.ascii as asc
import pandas as pd
from astropy.io.misc.hdf5 import read_table_hdf5
import matplotlib.pyplot as plt
from scipy.stats import norm

from IPython import embed

from dgf import diameterToAngle
from .dual_pop_metal import DualPopulationMetal
from .gaussian import GaussianFit


class Probability:
    def __init__(self, galaxy_name, config_file=None, obs_file=None):

        self.name = galaxy_name

        if config_file is None:
            config_file = f"../galaxies/gals/{galaxy_name}.json"

        if obs_file is None:
            obs_file = f"../galaxies/gals/{galaxy_name}_observation.hdf"

        obs = pd.read_hdf(obs_file)

        gal_prop_file = open(config_file)
        gal_prop = json.load(gal_prop_file)

        self.isochrone = asc.read(
            gal_prop["isochrone_path"], format="commented_header", header_start=13
        )

        # ['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe', 'logg', 'label', 'McoreTP', 'C_O', 'period0', 'period1', 'period2', 'period3', 'period4', 'pmode', 'Mloss', 'tau1m', 'X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess', 'Z', 'mbolmag', 'umag', 'gmag', 'rmag', 'imag', 'zmag']

        ## Galaxy parameters
        self.metal_0 = gal_prop["metal"]
        self.metal_err_0 = gal_prop["metal_err"]
        self.hrv_0 = gal_prop["HRV"]
        self.dhrv_0 = gal_prop["dHRV"]
        self.ra_0 = gal_prop["RA"] * u.deg
        self.dec_0 = gal_prop["DEC"] * u.deg
        self.dist = gal_prop["distance"] * u.kpc
        self.plummer = gal_prop["plummer_radius"] * u.pc

        ## Load stars from observation
        self.obs = pd.read_hdf(obs_file)

        self.prop = gal_prop
        # Index(['RA', 'DEC', 'HRV', 'dHRV', '[Fe/H]', 'r', 'g', 'color', 'member'], dtype='object')

    def cm(self, sigma=0.1, save_plot=False):

        r = self.obs["r"].to_numpy()
        color = self.obs["color"].to_numpy()
        isochrone_r = self.isochrone["rmag"].data
        isochrone_color = self.isochrone["gmag"].data - self.isochrone["rmag"].data

        if self.dist.value > 0:
            isochrone_r = 5 * np.log10(self.dist.to(u.pc).value) - 5 + isochrone_r

        dmin = []

        for i in np.arange(len(r)):
            distances = np.sqrt(
                (r[i] - isochrone_r) ** 2 + (color[i] - (isochrone_color)) ** 2
            )
            dmin.append(np.min(distances))

        dmin = np.array(dmin)

        prob = np.exp(-1 * (dmin ** 2) / (2 * sigma ** 2))

        if save_plot:
            fig, ax = plt.subplots(dpi=200)
            f = ax.scatter(color, r, c=prob, s=0.1)
            ax.set_ylabel(r"$r$")
            ax.set_xlabel(r"$g-r$")
            ax.invert_yaxis()
            plt.colorbar(f)
            plt.savefig(f"{self.name}_cm.png")

        return prob

    def radial(self, reff=None, save_plot=False):
        if reff is None:
            reff = (diameterToAngle(self.plummer, self.dist)).to(u.deg)

        print(f"Effective radius: {reff} [deg]")
        r = np.sqrt(
            ((self.obs["RA"].to_numpy() * u.deg) - self.ra_0) ** 2
            + ((self.obs["DEC"].to_numpy() * u.deg) - self.dec_0) ** 2
        )

        prob = np.exp(-(r ** 2) / (2 * reff ** 2))

        if save_plot:
            h = 10
            fig, ax = plt.subplots(dpi=200)
            f = ax.scatter(
                self.obs["RA"].to_numpy(), self.obs["DEC"].to_numpy(), c=prob, s=0.1
            )
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$\delta$")
            ax.set_xlim(
                self.ra_0.value - h * reff.value, self.ra_0.value + h * reff.value
            )
            ax.set_ylim(
                self.dec_0.value - h * reff.value, self.dec_0.value + h * reff.value
            )
            ax.invert_yaxis()
            plt.colorbar(f)
            plt.savefig(f"{self.name}_radial.png")

        return prob

    def dispersion(self, nwalkers=20, niter=1000, vwin=20, save_plot=False):
        fitter = GaussianFit(
            x_lims=(self.hrv_0 - vwin, self.hrv_0 + vwin),
            s_lims=(self.dhrv_0 - vwin, self.dhrv_0 + vwin),
            niter=niter,
            walkers=nwalkers,
        )

        _, _, _, sampler = fitter.run(
            self.obs["HRV"].to_numpy(),
            self.obs["dHRV"].to_numpy(),
            self.hrv_0,
            self.dhrv_0,
        )

        samples = sampler.flatchain

        res = fitter.get_results(samples)

        fit = {"HRV": res[0], "HRV_err": res[1], "dHRV": res[2], "dHRV_err": res[3]}

        if save_plot:
            fitter.plot(
                samples, labels=["v", r"$\sigma_v$"], save_to=f"{self.name}_vel.png",
            )

        probs = norm.pdf(self.obs["HRV"].to_numpy(), loc=fit["HRV"], scale=fit["dHRV"])

        return probs

    def dispersion_adv_project(self, mw_theta, rv_lim=[-200, 500]):

        theta = [
            1 - mw_theta[0],
            self.hrv_0,
            self.dhrv_0,
            self.metal_0,
            self.metal_err_0,
            mw_theta[1],
            mw_theta[2],
            mw_theta[3],
            mw_theta[4],
        ]

        fitter = DualPopulationMetal(
            theta,
            self.obs["HRV"].to_numpy(),
            self.obs["dHRV"].to_numpy(),
            self.obs["[Fe/H]"].to_numpy(),
            self.obs["d[Fe/H]"].to_numpy(),
        )

        popt = fitter.getOptimalValues(theta)

        fitter.project_model(theta, rvmin=np.min(rv_lim), rvmax=np.max(rv_lim))

    def dispersion_adv(self, mw_theta, nwalkers=20, niter=5000, save_plot=False):
        # theta
        # pgal
        # gal_hrv
        # gal_vsig
        # gal_feh
        # gal_fehsig
        # mw_hrv
        # mw_vsig
        # mw_feh
        # mw_fehsig

        theta = [
            1 - mw_theta[0],
            self.hrv_0,
            self.dhrv_0,
            self.metal_0,
            self.metal_err_0,
            mw_theta[1],
            mw_theta[2],
            mw_theta[3],
            mw_theta[4],
        ]

        fitter = DualPopulationMetal(
            theta,
            self.obs["HRV"].to_numpy(),
            self.obs["dHRV"].to_numpy(),
            self.obs["[Fe/H]"].to_numpy(),
            self.obs["d[Fe/H]"].to_numpy(),
        )
        popt = fitter.getOptimalValues(theta)
        _, _, _, sampler = fitter.run(nwalkers, popt, niters=niter, burnin=500)
        res = fitter.get_results()
        probs = norm.pdf(
            self.obs["HRV"].to_numpy(), loc=res["gal_hrv"], scale=res["gal_vsig"]
        )
        return probs

    def recovery_fn(self, cutoff=0.95, save_plot=False):
        prob_path = f"./data/prob_df_{self.name}.hdf"
        prob_df = pd.read_hdf(prob_path)

        cutoff_mask = prob_df.Prob > cutoff

        sample_size = sum(cutoff_mask)
        members = sum(prob_df[cutoff_mask].member == 1)
        total_members = sum(prob_df.member == 1)

        print(
            f"Sample size: {sample_size}, Members detected: {members}, Total members: {total_members}"
        )

        completeness = members / total_members
        purity = members / sample_size

        print(f"Completeness: {completeness}, Purity: {purity}")

        if save_plot:
            print("Making plot...")
            fig, axs = plt.subplots(2, 2, dpi=200)
            ((cm_ax, r_ax), (vd_ax, tot_ax)) = axs
            f = cm_ax.scatter(prob_df.RA, prob_df.DEC, c=prob_df["Prob.cm"], s=0.1)
            # cm_ax.invert_yaxis()
            r_ax.scatter(prob_df.RA, prob_df.DEC, c=prob_df["Prob.reff"], s=0.1)
            vd_ax.scatter(prob_df.RA, prob_df.DEC, c=prob_df["Prob.vel"], s=0.1)
            tot_ax.scatter(prob_df.RA, prob_df.DEC, c=prob_df["Prob"], s=0.1)
            plt.suptitle(
                rf"T $\geq$ {cutoff}, C: {np.round(completeness , 5)}, P: {np.round(purity , 5)}"
            )
            plt.colorbar(f)
            plt.savefig(f"./plots/prob_{cutoff}_{self.name}.png")

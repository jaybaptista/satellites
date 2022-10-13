import emcee
import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner
from scipy.stats import norm
from scipy.optimize import minimize


class DualPopulation:
    def __init__(self, p0, hrv, hrv_err, feh, feh_err):
        # theta_0 keys
        # pgal
        # gal_hrv
        # gal_vsig
        # mw_hrv
        # mw_vsig

        self.p0 = p0
        self.hrv = hrv
        self.hrv_err = hrv_err

        self.feh = feh
        self.feh_err = feh_err

    def lnprob(
        self,
        theta,
    ):

        (
            pgal,
            gal_hrv,
            gal_vsig,
            gal_feh,
            gal_fehsig,
            mw_hrv,
            mw_vsig,
            mw_feh,
            mw_fehsig,
        ) = theta

        gal_hrv_min, gal_hrv_max = [-500, 500]
        gal_vsig_min, gal_vsig_max = [1e-5, 100]
        gal_feh_min, gal_feh_max = [-2, 0]
        gal_fehsig_min, gal_fehsig_max = [1e-3, 0.4]
        mw_hrv_min, mw_hrv_max = [-300, 300]
        mw_vsig_min, mw_vsig_max = [10, 500]
        mw_feh_min, mw_feh_max = [-1, 1.5]
        mw_fehsig_min, mw_fehsig_max = [1e-3, 5]

        if (
            (pgal < 0)
            or (pgal > 1)
            or (gal_hrv > gal_hrv_max)
            or (gal_hrv < gal_hrv_min)
            or (gal_vsig > gal_vsig_max)
            or (gal_vsig < gal_vsig_min)
            or (gal_feh > gal_feh_max)
            or (gal_feh < gal_feh_min)
            or (gal_fehsig > gal_fehsig_max)
            or (gal_fehsig < gal_fehsig_min)
            or (mw_hrv > mw_hrv_max)
            or (mw_hrv < mw_hrv_min)
            or (mw_vsig > mw_vsig_max)
            or (mw_vsig < mw_vsig_min)
            or (gal_feh > gal_feh_max)
            or (gal_feh < gal_feh_min)
            or (mw_feh > mw_feh_max)
            or (mw_feh < mw_feh_min)
            or (mw_fehsig > mw_fehsig_max)
            or (mw_fehsig < mw_fehsig_min)
        ):
            return -np.inf

        lgal_hrv = norm.logpdf(
            self.hrv, loc=gal_hrv, scale=np.sqrt(self.hrv_err**2 + gal_vsig**2)
        )
        lmw_hrv = norm.logpdf(
            self.hrv, loc=mw_hrv, scale=np.sqrt(self.hrv_err**2 + mw_vsig**2)
        )
        lgal_feh = norm.logpdf(
            self.feh, loc=gal_feh, scale=np.sqrt(self.feh_err**2 + gal_fehsig**2)
        )
        lmw_feh = norm.logpdf(
            self.feh, loc=mw_feh, scale=np.sqrt(self.feh_err**2 + mw_fehsig**2)
        )

        lgal = np.log(pgal) + lgal_hrv + lmw_feh
        lmw = lmw_hrv + lgal_feh
        ltot = np.logaddexp(lgal, np.log(1 - pgal) + lmw)
        val = np.sum(ltot)

        if val > 1e10:
            print("Infinity warning...", val)

        return val

    def project_model(self, theta, rvmin=-200, rvmax=500, fehmin=-3, fehmax=2):

        (
            pgal,
            gal_hrv,
            gal_vsig,
            gal_feh,
            gal_fehsig,
            mw_hrv,
            mw_vsig,
            mw_feh,
            mw_fehsig,
        ) = theta

        vels = np.linspace(rvmin, rvmax, 1000)
        fehs = np.linspace(fehmin, fehmax, 500)

        prv_gal = pgal * norm.pdf(vels, loc=gal_hrv, scale=gal_vsig)
        prv_mw = (1 - pgal) * norm.pdf(vels, loc=mw_hrv, scale=mw_vsig)

        pfeh_gal = pgal * norm.pdf(fehs, loc=gal_feh, scale=gal_fehsig)
        pfeh_mw = (1 - pgal) * norm.pdf(fehs, loc=mw_feh, scale=mw_fehsig)

        fig, ax = plt.subplots(1, 2, dpi=150)

        ax[0].hist(self.hrv, bins="auto", color="grey", density=True)

        ax[0].plot(vels, (prv_gal + prv_mw), c="k", lw=3, label="Total")

        ax[0].plot(vels, prv_gal, c="b", lw=2, label="Satellite")

        ax[0].plot(vels, prv_mw, c="orange", lw=2, label="MW")

        ax[0].legend()

        #####

        ax[1].hist(self.feh, bins="auto", color="grey", density=True)

        ax[1].plot(fehs, (pfeh_gal + pfeh_mw), c="k", lw=3, label="Total")

        ax[1].plot(fehs, pfeh_gal, c="b", lw=2, label="Satellite")

        ax[1].plot(fehs, pfeh_mw, c="orange", lw=2, label="MW")

        ax[1].legend()

        plt.show()

    def optimizer(self, theta):
        return -self.lnprob(theta)

    def getOptimalValues(self, theta=None):

        if theta is None:
            theta = self.p0

        res = minimize(self.optimizer, theta, method="Nelder-Mead")
        print(res.message)
        return res.x

    def run(self, nwalkers, p0=None, burnin=200, niters=5000):

        if p0 is None:
            p0 = self.p0

        ndim = len(p0)

        p0 = [p0 + 1e-6 * np.random.rand(ndim) for walker in range(nwalkers)]

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob)

        p0, _, _ = sampler.run_mcmc(p0, burnin, progress=True)
        sampler.reset()

        pos, prob, state = sampler.run_mcmc(p0, niters, progress=True)

        self.sampler = sampler

        return pos, prob, state, sampler

    def plot(self, labels=None):
        chain = self.sampler.flatchain

        # theta_0 keys
        # pgal
        # gal_hrv
        # gal_vsig
        # mw_hrv
        # mw_vsig

        if labels == None:
            labels = ["pgal", "gal_hrv", "gal_vsig", "mw_hrv", "mw_vsig"]

        fig = corner.corner(
            chain,
            show_titles=True,
            labels=labels,
            plot_datapoints=True,
            quantiles=[0.16, 0.5, 0.84],
            dpi=150,
        )
        fig.tight_layout()

        plt.show()

    def get_results(self):

        self.results = {
            "pgal": np.percentile(self.sampler.flatchain[:, 0], 50),
            "hrv_gal": np.percentile(self.sampler.flatchain[:, 1], 50),
            "vsig_gal": np.percentile(self.sampler.flatchain[:, 2], 50),
            "hrv_mw": np.percentile(self.sampler.flatchain[:, 3], 50),
            "vsig_mw": np.percentile(self.sampler.flatchain[:, 4], 50),
        }

        return self.results

    def pdf(self, hrv):
        return norm.pdf(
            hrv, loc=self.results["hrv_gal"], scale=self.results["vsig_gal"]
        )

import emcee
import numpy as np
import emcee
import matplotlib.pyplot as plt
import corner


class GaussianFit:
    def __init__(self, x_lims, s_lims, burn_in=400, niter=4000, walkers=20):
        """
        Class that handles MCMC sampling of a Gaussian distribution

        Parameters
        ----------
        x_lims: tuple
            limits on mean
        s_lims: tuple
            limits on dispersion
        burn_in: int (400)
            burn-in steps
        niter: int (4000)
            num of iters
        walkers: int (20)
            num of walkers
        """
        self.x_upper_lim = np.max(x_lims)
        self.x_lower_lim = np.min(x_lims)
        self.s_upper_lim = np.max(s_lims)
        self.s_lower_lim = np.min(s_lims)

        self.burn_in = burn_in
        self.niter = niter
        self.walkers = walkers

    def lnlike(self, theta, x_star, s_star):
        x, s = theta
        N = len(x_star)
        error = (s**2 + s_star**2) ** (1 / 2)
        model = (
            -0.5 * N * np.log(2 * np.pi)
            - (np.sum(np.log(error)))
            - (0.5 * np.sum((x_star - x) ** 2 / (error**2)))
        )

        return model

    def lnprior(self, theta):
        x, s = theta
        if (
            (self.x_upper_lim > x)
            & (x > self.x_lower_lim)
            & (s > self.s_lower_lim)
            & (s < self.s_upper_lim)
        ):
            return 0.0
        else:
            return -np.inf

    def lnprob(self, theta, x_star, s_star):
        lp = self.lnprior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.lnlike(theta, x_star, s_star)

    def init_walkers(self, x_guess, s_guess):
        ndim, nwalkers = 2, self.walkers

        p0 = [
            (x_guess, s_guess) + 1e-6 * np.random.rand(ndim)
            for walker in range(nwalkers)
        ]

        return ndim, nwalkers, p0

    def run(self, x_star, s_star, x_guess, s_guess):
        ndim, nwalkers, p0 = self.init_walkers(x_guess, s_guess)
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.lnprob, args=(x_star, s_star)
        )
        p0, _, _ = sampler.run_mcmc(p0, self.burn_in, progress=True)
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(p0, self.niter, progress=True)
        return pos, prob, state, sampler

    def get_results(self, samples, rel=True):
        x = np.percentile(samples[:, 0], 50)
        x_err = np.percentile(samples[:, 0], [16, 84])
        s = np.percentile(samples[:, 1], 50)
        s_err = np.percentile(samples[:, 1], [16, 84])

        if rel:
            x_err -= x
            s_err -= s

        return x, x_err, s, s_err

    def plot(self, chain, labels=None, save_to=None):
        if labels == None:
            labels = ["x", "s"]
        fig = corner.corner(
            chain,
            show_titles=True,
            labels=labels,
            plot_datapoints=True,
            quantiles=[0.16, 0.5, 0.84],
            dpi=150,
        )
        fig.tight_layout()

        if save_to != None:
            plt.savefig(save_to)
        else:
            plt.show()

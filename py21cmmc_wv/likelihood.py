"""
A module defining CosmoHammer likelihoods for addition into the standard 21cmMC structure.
"""

import numpy as np
from powerbox.dft import fft
from powerbox.tools import angular_average_nd
from py21cmmc.mcmc import core, likelihood
from scipy import special as sp

from .morlet import morlet_transform_c
from .util import lognormpdf


class LikelihoodWaveletsMorlet(likelihood.LikelihoodBaseFile):
    """
    This likelihood is based on Morlet wavelets, as found in eg. Trott+2016.

    However, this likelihood has no *instrument* involved -- thus it can compute directly in k-space.

    """
    required_cores = [core.CoreLightConeModule]

    def __init__(self, bins=None, model_uncertainty=0.15, **kwargs):
        super().__init__(**kwargs)

        self.bins = bins
        self.model_uncertainty = model_uncertainty

    def computeLikelihood(self, model):

        model = model[0]

        x = self.data[0]['wavelets']
        mu = model['wavelets']

        # TODO: data/model should probably be transposed in some way.
        return lognormpdf(
            x=x.reshape((-1, x.shape[-1])),
            mu=mu.reshape((-1, mu.shape[-1])),
            cov=self.compute_covariance(model['wavelets'], model['kpar'], model['centres'])
        )

    @staticmethod
    def compute_wavelets(lightcone, bins):

        # First get "visibilities"
        vis, kperp = fft(lightcone.brightness_temp, L=lightcone.user_params.BOX_LEN, axes=(0, 1))

        # vis has shape (HII_DIM, HII_DIM, lightcone_dim)

        # Do wavelet transform
        wvlts, kpar, _ = morlet_transform_c(vis.T, lightcone.lightcone_coords)

        # wvlts has shape (len(kpar) + vis.T.shape) corresponding to (eta, nu_c, u,v)

        # Now square it...
        wvlts = np.abs(wvlts) ** 2

        # Determine a nice number of bins.
        if bins is None:
            bins = int((np.product(kperp.shape) * len(kpar)) ** (1. / 3.) / 2.2)

        # And angularly average
        wvlts, k = angular_average_nd(
            wvlts.transpose((2, 3, 0, 1)),
            list(kperp) + [kpar, lightcone.lightcone_coords],
            n=3, bins=bins, bin_ave=False, get_variance=False
        )

        return wvlts, k, lightcone.lightcone_coords

    def compute_covariance(self, wvlts, kpar, dist):

        # wvlts (from compute_wavelets) has shape (Nkperp, Nkpar, Nz).
        # only the Nz will correlate (on the assumption that power is independent in each mode).

        cov = []
        D = np.abs(np.add.outer(dist, -dist))

        for ix in range(wvlts.shape[0]):
            for ikp in range(wvlts.shape[1]):
                wvlt = wvlts[ix, ikp]
                thiscorr = (1 - sp.erf(kpar[ikp] * D / (2 * np.sqrt(2)))) ** 2
                cov.append(thiscorr * self.model_uncertainty ** 2 * np.outer(wvlt, wvlt))

        return cov

    def reduce_data(self, ctx):
        wvlt, k, centres = self.compute_wavelets(ctx.get("lightcone"), self.bins)
        return [dict(wavelets=wvlt, k=k, centres=centres)]

    @property
    def lightcone_module(self):
        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, self.required_cores[0]):
                return m

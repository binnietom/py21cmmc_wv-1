"""
A module defining CosmoHammer likelihoods for addition into the standard 21cmMC structure.
"""

from py21cmmc.mcmc import core, likelihood
from .morlet import morlet_transform
from powerbox.tools import angular_average_nd
from powerbox.dft import fft
import numpy as np


class LikelihoodWaveletsMorlet(likelihood.LikelihoodBase):
    """
    This likelihood is based on Morlet wavelets, as found in eg. Trott+2016.

    However, this likelihood has no *instrument* involved -- thus it can compute directly in k-space.

    """
    required_cores = [core.CoreLightConeModule]

    def __init__(self, datafile, n_kperp=None, **kwargs):
        super().__init__(datafile, **kwargs)
        # Determine a nice number of bins.
        self.n_kperp = n_kperp
        self.datafile = datafile

    def setup(self):
        super().setup()

    def computeLikelihood(self, ctx, storage):
        model = self.simulate(ctx)

        storage.update(**model)

        var = self.compute_variance(model['wavelets'])
        return -np.sum((model['wavelets'] - self.data['wavelets'])**2/var)

    @staticmethod
    def compute_wavelets(lightcone, n_kperp):

        # First get "visibilities"
        vis, kperp = fft(lightcone.brightness_temp, L=lightcone.user_params.HII_DIM, axes=(0, 1))

        # Determine a nice number of bins.
        if n_kperp is None:
            n_kperp = int(np.product(kperp.shape) ** (1. / 2.)/2.2)

        # Do wavelet transform
        wvlts, kpar, _ = morlet_transform(vis, lightcone.lightcone_coords)

        # Now square it...
        wvlts = np.abs(wvlts)**2

        # And angularly average
        wvlts, kperp = angular_average_nd(wvlts, list(kperp)+[lightcone.lightcone_coords, kpar], n=2, bins=n_kperp, bin_ave=False)

        return wvlts, kperp, kpar, lightcone.lightcone_coords

    def compute_covariance(self, nrealisations=200):
        wvlt = []
        for i in range(nrealisations):
            wvlt.append(self.simulate(self.default_ctx)['wavelets'])

        # Now get covariance of them all...
        # Only *co*-vary in the "centres" direction.
        # Wavelets has shape (kperp, centres, kpar).
        cov = np.cov(wvlt.transpose(0,2,1))
        return (0.15*wvlt)**2

    def simulate(self, ctx):
        wvlt, kperp, kpar, centres = self.compute_wavelets(ctx.get("lightcone"), self.n_kperp)
        return dict(wavelets=wvlt, kperp=kperp, kpar=kpar, centres=centres)
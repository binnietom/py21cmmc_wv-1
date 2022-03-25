"""
A module defining CosmoHammer likelihoods for addition into the standard 21cmMC structure.
"""

import numpy as np
from powerbox.dft import fft
from powerbox.tools import angular_average_nd
from py21cmmc import core, likelihood
from scipy import special as sp
import math as m

from .morlet import morlet_transform_c
from .util import lognormpdf
from .util import loaded_lognormpdf

class LikelihoodWaveletsMorlet(likelihood.LikelihoodBaseFile):
    """
    This likelihood is based on Morlet wavelets, as found in eg. Trott+2016.

    However, this likelihood has no *instrument* involved -- thus it can compute directly in k-space.

    """
    required_cores = [core.CoreLightConeModule]

    def __init__(self, bins=None, nchunks=1, model_uncertainty=1., stride=1, cov='est', BHF=False, **kwargs):
        super().__init__(**kwargs)

        self.bins = bins
        self.model_uncertainty = model_uncertainty
        self.stride = stride
        self.nchunks = nchunks
        self.cov = cov
        self.BHF = BHF
        #Covariance will be estimated if 'est'; to load the covariance put cov='the path and file name'(-.npy).

    def computeLikelihood(self, model):
        """
        old likelihood function is inside the loop.
        Calculates the likelihood in the same way as 21cmmc
        we have w,k,k,c in a dictionary per chunk.
        """
        lnL=0
        #print("\n got to likelihood \n")
        for i in range(self.nchunks): #loop through chunks# stack likelihood.
            mu = model[i]['wavelets']
            x = self.data[i]['wavelets']

            #mu = model['wavelets']  ##don't overwrite model!
            if (self.cov == 'est'):
                covariance = self.compute_covariance(mu, model[i]['kpar'], model[i]['centres'])

                L = lognormpdf(
                    x=x.reshape((-1, x.shape[-1])),
                    mu=mu.reshape((-1, mu.shape[-1])),
                    cov=covariance,
                )
            if (self.cov != 'est'):
                #print(f"loading cov {i}")
                if (self.nchunks == 3):
                    covariance = np.load(f"{self.cov}chunk_{i}.npy", allow_pickle=True)
                if (self.nchunks == 1):
                    covariance = np.load(f"{self.cov}.npy", allow_pickle=True)
                #print("loaded cov")
                L = loaded_lognormpdf( #lognormpdf2(
                    x=x.reshape((-1, x.shape[-1])),
                    mu=mu.reshape((-1, mu.shape[-1])),
                    cov=covariance,  #####. i for chunk. but we just load each time (better for memory this way)
                )

            if (m.isnan(L)):
                print( 'Nan lnL ')
                return np.inf
            else:
                lnL += L
            #print("\n likelihood for chunk ", i, " is ", L)
        #print("\n total Likelihood: ", lnL, "\n")
        return lnL

    @staticmethod
    def compute_mps(lightcone, bins=None, nthreads=None, nchunks=1, stride=1, integral_width=5, BHF=False):
        """
        computing the mps on chunks of the lightcone - stacked in an array just like 21cmmc does for the FPS
        This is structured so that the old compute_mps function is now compute_mt so that any compute_mps references don't break
        If no chunks inputted, nchunks = 1, the whole lightcone is one chunk (equivalent to no chunking).
        """
        #print("\n chunks: ", nchunks)
        data = []
        chunk_indices = list(range(0,lightcone.n_slices,round(lightcone.n_slices / nchunks),))

        if len(chunk_indices) > nchunks:
            chunk_indices = chunk_indices[:-1]
        chunk_indices.append(lightcone.n_slices)

        #print("\n entering chunk loop \n")
        for i in range(nchunks):
            start = chunk_indices[i]

            try:
                end = chunk_indices[i + 1]
            except IndexError:
                if i == nchunks-1 and nchunks >1:
                    end = lightcont.n_slices
                else: break
            chunklen = (end - start) * lightcone.cell_size

            #wvlts, kperp_mod, kpar, centres = compute_mt(lightcone.lightcone_coords[start:end])
            # ^this was buggy, just put old-code here Tom - wanted to avoid passing lightcones
            #print("\n chunk:", i," starting visibilities  \n")
            #print(f"\n visibility inputs: \n Tb: {lightcone.brightness_temp[:, :, start:end].shape} , \n Box:  {lightcone.user_params.BOX_LEN} \n \n")

            # First get "visibilities" with shape (HII_DIM, HII_DIM, lightcone_dim)
            vis, kperp = fft(lightcone.brightness_temp[:, :, start:end], L=lightcone.user_params.BOX_LEN, axes=(0, 1))

            #print("\n chunk:", i," visibilities done  \n")
            centres = lightcone.lightcone_coords[start:end]
            centres = centres[::stride] ## so if stride = 1 and nchunks = 1 centres = array of every pixel along the lightcone.
            #print("\n chunk:", i," entering MPS \n")
            # Do wavelet transform
            print(f'Using BlackmanHarris Filter? {BHF}')
            wvlts, kpar, _ = morlet_transform_c(vis.T, centres, convergence_extent=integral_width, nthreads=nthreads, stride = stride, BHF=BHF)
            #print("\n chunk:", i," MPS done  \n")
            # Now remove complex.
            wvlts = np.abs(wvlts) ** 2
            # Determine a nice number of bins >70 or wavelet form changes.
            if bins is None:
                bins = int(49) ##why?

            #print("\n chunk:", i," averaging  \n")
            #before angularly average wavelets (transpose) has shape (kperp, kperp, kparr, centres)
            wvlts, kperp_mod = angular_average_nd(
                wvlts.transpose((2, 3, 0, 1)),
                list(kperp) + [kpar, centres],
                n=2, bins=bins, bin_ave=False, get_variance=False
            )
            #n=2 takes angular average of first two inputs, which when wavelet is transposed is k_par,k_par.
            #wvlts now has shape (kperpmod, k_par, centres)
            #print("\n chunk:", i," appending dictionary with averaged MPS  \n")
            data.append({"wavelets": wvlts, "kperp_mod": kperp_mod, "kpar": kpar, "centres": centres})
        return np.array(data)

    def compute_covariance(self, wvlts, kpar, dist):
        """
        wvlts (from compute_mps) has shape (N_kperp, Nk_parr, N_xz).
        only the z direction kparr will correlate (on the assumption that power is independent in each k_perp mode).
        """
        cov = []
        D = np.abs(np.add.outer(dist, -dist))

        for ix in range(wvlts.shape[0]):
            for ikp in range(wvlts.shape[1]):
                wvlt = wvlts[ix, ikp]
                thiscorr = np.exp((-1/4)*kpar[ikp]*kpar[ikp]*D)
                cov.append(thiscorr * (self.model_uncertainty ** 2) * np.outer(wvlt, wvlt))
        return np.array(cov)

    @staticmethod
    def comp_cov_check(mw_realisation):
        """
        Calculating the Covariance directly from multiple wavelets
        expects [N, [wavelet(kperp, kparr, centres), kperpM,  kparr, centres]] from N different lightcone realisations
        returns kperp x kparr covariance matrices (centres, centres) measured from N realisations
        """
        covs = []
        centres = mw_realisation[0,3] #lc coords all the same, just take 1st.
        lenc = len(centres)
        wvlts = []
        for ireal in range(len(mw_realisation)):
            wvlts.append(mw_realisation[ireal,0]) #get array of just the  wavelets
        wvlts = np.array(wvlts)

        for ikperp in range(len(wvlts[0,:,0,0])):
            cov = []
            for ikparr in range(len(wvlts[0,0,:,0])):
                wvlt = wvlts[:,ikperp, ikparr, :] #[realisations, centres] per [kperp, kparr]
                c = np.zeros((lenc,lenc))
                for xi_c in range(lenc):
                    for xj_c in range(lenc):
                        c[xi_c,xj_c] = np.mean(np.dot(wvlt[:,xi_c],wvlt[:,xj_c])) - np.mean(wvlt[:,xi_c])*np.mean(wvlt[:,xj_c])
                cov.append(c)
            covs.append(cov)
        return np.array(covs)/len(mw_realisation) # Divide by N realisations!

    @staticmethod
    def cov_FPS_comparison(fps_realisation):
        """
        Calcualting the covariance between N realisations of the FPS.
        Similar to comp_cov_check but for comparisons with to FPS
        expects [N, [nchunks, {"k":k, "delta":delta}]]
        returns nchunks covariance matrices (k, k) measured from N realisations.
        """
        covs = []
        nchunks = len(fps_realisation[0]) #same for each N
        #start cov loops
        for ichunk in range(nchunks):
            ks = fps_realisation[0][ichunk]["k"] #same for each N
            lenk = len(ks)
            FPSs = [] #get array of just FPS
            for i in range(len(fps_realisation)):
                FPSs.append(fps_realisation[i][ichunk]["delta"])
            FPSs = np.array(FPSs)
            cov = np.zeros((lenk,lenk))
            for ik in range(lenk):
                for jk in range(lenk):
                    cov[ik,jk] = np.mean(np.dot(FPSs[:,ik],FPSs[:,jk])) - np.mean(FPSs[:,ik])*np.mean(FPSs[:,jk])
                    if (m.isnan(cov[ik,jk])): #when an array has nan's they are all nans - set to zero for plotting.
                        cov[ik,jk] = 0
                        #print(" FPSs array has nans (chunk, k)  (", ichunk," ,  " , ik , ") ")
            covs.append(cov)
        return np.array(covs)/len(fps_realisation)


    def reduce_data(self, ctx):
        #wvlt, kperp_mod, kpar,  centres = self.compute_mps(ctx.get("lightcone"), self.bins, stride = self.stride)
        #return [dict(wavelets=wvlt, kperp_mod=kperp_mod, kpar=kpar, centres=centres)]
        return self.compute_mps(ctx.get("lightcone"), self.bins, nchunks = self.nchunks, stride = self.stride)

    @property
    def lightcone_module(self):
        for m in self.LikelihoodComputationChain.getCoreModules():
            if isinstance(m, self.required_cores[0]):
                return m

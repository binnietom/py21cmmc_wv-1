from py21cmmc_wv.likelihood import LikelihoodWaveletsMorlet
from py21cmmc.mcmc.mcmc import run_mcmc
from py21cmmc.mcmc.core import CoreLightConeModule
import os
import sys

model_name = "runthrough_test2"

core = CoreLightConeModule(
    redshift=7.0,
    max_redshift=8.0,
    user_params=dict(
        HII_DIM=30,
        BOX_LEN=60.0
    ),
    regenerate=False
)

likelihood = LikelihoodWaveletsMorlet(
    datafile = "data/runthrough_test2.npz",
    nrealisations=50
)

chain = run_mcmc(
    core, likelihood,
    datadir='data',
    model_name=model_name,
    params = dict(
        HII_EFF_FACTOR=[30.0, 10., 50.0, 3.0],
        ION_Tvir_MIN = [4.7, 2, 8, 0.1]
    ),
    walkersRatio=2,
    burninIterations=0,
    sampleIterations=2,
    threadCount=6,
    continue_sampling=False
)

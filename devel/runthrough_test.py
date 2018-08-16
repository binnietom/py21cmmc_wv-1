from py21cmmc_wv.likelihood import LikelihoodWaveletsMorlet
from py21cmmc.mcmc.mcmc import run_mcmc
from py21cmmc.mcmc.core import CoreLightConeModule
import os
import sys

model_name = "runthrough_test_large"

core = CoreLightConeModule(
    redshift=7.0,
    max_redshift=8.0,
    user_params=dict(
        HII_DIM=50,
        BOX_LEN=100.0
    ),
    regenerate=False
)

likelihood = LikelihoodWaveletsMorlet(
    datafile = "data/runthrough_test_large.npz",
)

chain = run_mcmc(
    core, likelihood,
    datadir='data',
    model_name=model_name,
    params = dict(
        HII_EFF_FACTOR=[30.0, 10., 50.0, 3.0],
        ION_Tvir_MIN = [4.7, 2, 8, 0.1]
    ),
    walkersRatio = 6,
    burninIterations=0,
    sampleIterations=200,
    threadCount=6,
    continue_sampling=False
)

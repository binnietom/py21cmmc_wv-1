from py21cmmc_wv import morlet
import numpy as np

bw = 50.0
numin = 130.0
N = 736
nu = np.arange(N) * bw/N + numin
mid = (nu[0] + nu[-1])/2

spectrum = np.exp(-(nu-mid)**2/ (2*4.0**2))

trnsc, fc, _ = morlet.morlet_transform_c(spectrum, nu)
trnsc = np.abs(trnsc)**2

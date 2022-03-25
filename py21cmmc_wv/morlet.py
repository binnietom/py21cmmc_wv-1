import ctypes
import glob
import os

from scipy.signal.windows import blackmanharris
#from scipy.fft import fft, fftshift

import numpy as np
from multiprocessing import cpu_count

# Build the extension function (this should be negligible performance-wise)
fl = glob.glob(os.path.join(os.path.dirname(__file__), "ctransforms*"))[0]


def morlet_transform_c(data, nu, convergence_extent=10.0, fourier_b = 1,
                       vol_norm=False, nthreads=None, BHF=False):
    """
    Perform a Morlet Transform using underlying C code.

    Parameters
    ----------
    data : array_like, SHAPE=[N_NU, ...]
        The visibility data on which to perform the Morlet Transform. The
        transform itself occurs over the first axis.
    nu : array_like, SHAPE=[N_NU]
        The frequencies
    convergence_extent : float, optional
        How many sigma to integrate the Morlet kernel
    fourier_b : float, optional
        Defines the Fourier convention.
    vol_norm : bool, optional
        Whether to apply a volume normalisation so that different eta
        have the same expected power.
    nthreads : int, optional
        Number of threads to use in transform. Default is all of them.
    BHF : bool, optional
        Use a BlackmanHarris Filter (True) instead of a Gaussian (False) in the wavelets.

    Returns
    -------
    complex array, SHAPE=[N_ETA, N_NU, ...]
        The output transformed visibilities.
    """


    if BHF == False:
        morlet = ctypes.CDLL(fl).cmorlet
        morlet.argtypes = [
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_double, ctypes.c_double,
            np.ctypeslib.ndpointer("complex128", flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer("complex128", flags="C_CONTIGUOUS"),
        ]
    if BHF == True:
        morlet = ctypes.CDLL(fl).BlackmanHarris_cmorlet
        morlet.argtypes = [
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_double,
            np.ctypeslib.ndpointer(ctypes.c_double flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer("complex128", flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer("complex128", flags="C_CONTIGUOUS"),
        ]

    if nthreads is None:
        nthreads = cpu_count()

    assert nthreads <= cpu_count()

    # Get data into shape (everything_else, len(nu))
    orig_shape = data.shape
    n_nu = orig_shape[0]
    n_data = int(np.product(orig_shape[1:]))

    assert n_nu == len(nu)
    data = np.ascontiguousarray(data.flatten())

    dnu = (nu.max() - nu.min()) / (n_nu - 1)
    L = n_nu * dnu

    eta = np.arange(1, n_nu / 2) / L
    n_eta = len(eta)

    out = np.zeros(n_data * n_nu * n_eta, dtype=np.complex128)

    if BHF == False:
        morlet(n_data, n_nu, n_eta, float(convergence_extent), fourier_b,
            data, nu, eta, nthreads, out)

    if BHF == True:
        #Do we want the window or the frequency response? - I think frequency response, transform.c is all nu and eta.
        #BHF_FFT = fft(window, 2048) / (len(window)/2.0)
        #freq = np.linspace(-0.5, 0.5, len(A))
        #response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
        morlet(n_data, n_nu, n_eta, float(convergence_extent), blackmanharris(n_nu),
            data, nu, eta, nthreads, out)

    if vol_norm:
        norm = np.sqrt(np.abs(eta)) * dnu * np.pi ** (-1. / 4)
    else:
        norm = dnu
    print(out[350])

    out = norm * out.reshape((len(eta),) + orig_shape) # Make the array.

    return out, eta, nu


def morlet_transform(data, t, fourier_b=1):
    """
    Pure python version

    In current configuration, data can be N-dimensional, but must only be transformed over the *last* dimension.

    t here corresponds to 'nu' in terms of the sky.
    """
    # Get data into shape (everything_else, len(nu))
    orig_shape = data.shape
    n = orig_shape[-1]
    data = data.reshape((-1, n))  # Make it 2D

    dt = (t.max() - t.min()) / (n - 1)
    L = n * dt

    f = np.arange(1, n / 2) / L

    ans = np.zeros(data.shape + (len(f),), dtype=np.complex128)

    for i, d in enumerate(data):
        reduced = np.outer(np.add.outer(t, -t), f).reshape((len(t), len(t), len(f))).T
        ans[i] = np.sum(np.exp(-reduced ** 2 / 2) * np.exp(fourier_b * reduced * 1j) * d.T, axis=-1).T

    # for i,tt in enumerate(t):
    #     reduced = np.outer(tt - t, f)
    #     ans += np.exp(-reduced**2/2) * np.exp(2*np.pi*reduced*1j) * data[..., i]

    #    rl, im = morlet(np.real(data), np.imag(data), t, t, f) # Here data is complex.

    norm = np.sqrt(np.abs(f)) * dt * np.pi ** (-1. / 4)
    return (norm * ans).reshape(orig_shape + (len(f),)), f, t

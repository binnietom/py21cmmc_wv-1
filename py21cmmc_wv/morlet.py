import numpy as np
# from .transforms import morlet
import os
import ctypes
import glob


def morlet_transform_c(data, nu, convergence_extent=10.0):
    """
    Fortran version

    In current configuration, data can be N-dimensional, but must only be transformed over the *last* dimension.

    t here corresponds to 'nu' in terms of the sky.
    """
    # Build the extension function (this should be negligible performance-wise)
    fl = glob.glob(os.path.join(os.path.dirname(__file__), "ctransforms*"))[0]

    if data.dtype.name == "float64":
        morlet = ctypes.CDLL(fl).morlet
        morlet.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
            np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer("complex128", flags="C_CONTIGUOUS"),
        ]
    else:
        morlet = ctypes.CDLL(fl).cmorlet
        morlet.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double,
            np.ctypeslib.ndpointer("complex128", flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer("complex128", flags="C_CONTIGUOUS"),
        ]

    # Get data into shape (everything_else, len(nu))
    orig_shape = data.shape
    n_nu = orig_shape[-1]
    n_data = int(np.product(orig_shape[:-1]))

    data = data.flatten()

    dnu = (nu.max() - nu.min()) / (n_nu-1)
    L = n_nu*dnu

    eta = np.arange(1, n_nu/2) / L
    n_eta = len(eta)

    print(n_data, n_nu, n_eta, n_data*n_eta*n_nu*16/1024**3)
    out = np.zeros(n_data * n_nu * n_eta, dtype=np.complex128)

    morlet(n_data, n_nu, n_eta, float(convergence_extent), np.ascontiguousarray(data), nu, eta, out)

    norm = np.sqrt(np.abs(eta))*dnu*np.pi**(-1./4)
    return  norm * out.reshape(orig_shape+(len(eta),)), eta, nu


def _morlet_transform(data, t):
    """
    Fortran version

    In current configuration, data can be N-dimensional, but must only be transformed over the *last* dimension.

    t here corresponds to 'nu' in terms of the sky.
    """
    # Get data into shape (everything_else, len(nu))
    orig_shape = data.shape
    n = orig_shape[-1]
    data = data.reshape((-1, n))  # Make it 2D

    dt = (t.max() - t.min()) / (n-1)
    L = n*dt

    f = np.arange(1, n/2) / L

    rl, im = morlet(np.real(data), np.imag(data), t, t, f) # Here data is complex.

    norm = np.sqrt(np.abs(f))*dt*np.pi**(-1./4)
    return  norm * (rl + im * 1j).reshape(orig_shape+(len(f),)), f, t


def morlet_transform(data, t):
    """
    Pure python version

    In current configuration, data can be N-dimensional, but must only be transformed over the *last* dimension.

    t here corresponds to 'nu' in terms of the sky.
    """
    # Get data into shape (everything_else, len(nu))
    orig_shape = data.shape
    n = orig_shape[-1]
    data = data.reshape((-1, n))  # Make it 2D

    dt = (t.max() - t.min()) / (n-1)
    L = n*dt

    f = np.arange(1, n/2) / L

    ans = np.zeros(data.shape+(len(f),), dtype=np.complex128)

    for i,d in enumerate(data):
        reduced = np.outer(np.add.outer(t,-t), f).reshape((len(t), len(t), len(f))).T
        ans[i] = np.sum(np.exp(-reduced**2/2) * np.exp(2*np.pi*reduced*1j) * d.T, axis=-1).T

    # for i,tt in enumerate(t):
    #     reduced = np.outer(tt - t, f)
    #     ans += np.exp(-reduced**2/2) * np.exp(2*np.pi*reduced*1j) * data[..., i]

#    rl, im = morlet(np.real(data), np.imag(data), t, t, f) # Here data is complex.

    norm = np.sqrt(np.abs(f))*dt*np.pi**(-1./4)
    return  (norm * ans).reshape(orig_shape + (len(f),)), f, t
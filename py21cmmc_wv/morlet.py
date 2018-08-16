import numpy as np
from .transforms import morlet


def morlet_transform(data, t):
    """
    In current configuration, data can be N-dimensional, but must only be transformed over the *last* dimension.

    t here corresponds to 'nu' in terms of the sky.
    """
    # Get data into shape (everything_else, len(nu))
    orig_shape = data.shape
    n = orig_shape[-1]
    data = data.reshape((-1, n))  # Make it 2D

    dt = (t.max() - t.min()) / (n-1)
    L = n*dt

    f = np.arange(1, n / 2) / L

    rl, im = morlet(np.real(data), np.imag(data), t, t, f) # Here data is complex.

    norm = np.sqrt(np.abs(f))*dt*np.pi**(-1./4)
    return  norm * (rl + im * 1j).reshape(orig_shape+(len(f),)), f, t

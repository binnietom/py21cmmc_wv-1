#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include "logger.h"

//void morlet(int ndata, int n_nu, int n_eta, double conv_ext, double fourier_b,
//            double *data, double *nu, double *eta, double complex *out){
//    // Discrete Morlet Wavelet transform, using Morlet basis from Goupillaud 1984 (Eq. 5, 6 - with b=2pi)
//
//    int ix, jnuc,jeta, jnu, thisn;
//    double exponent, mag, extent, dt;
//
//    dt = nu[1] - nu[0];
//
//    double sqrt2 = sqrt(2.0);
//    int index = 0;
//
//    for (ix=0;ix<ndata;ix++){
//        for (jnuc=0;jnuc<n_nu;jnuc++){
//            for (jeta=0; jeta<n_eta;jeta++){
//                extent = 1/(eta[jeta]*sqrt2);
//                thisn = ceil(conv_ext*extent/dt);
//
//                for (jnu=fmax(0, jnuc-thisn); jnu<fmin(jnuc+thisn, n_nu); jnu++){
//                    exponent = eta[jeta]*(nu[jnu] - nu[jnuc]);
//                    out[index] += data[ix*n_nu + jnu]*cexp(-exponent*(exponent/2 + fourier_b*I));
//                }
//                index++;
//            }
//        }
//    }
//}

inline int max(int a, int b) {
    return a > b ? a : b;
}

inline int min(int a, int b) {
    return a < b ? a : b;
}

void cmorlet(unsigned int ndata, unsigned int n_nu, unsigned int n_eta,
             double conv_ext, double fourier_b,
             double complex *data, double *nu, double *eta, int nthreads,
             double complex *out){
    /*
        Discrete Morlet Wavelet transform
        =================================

        Uses Morlet basis from Goupillaud 1984 (Eq. 5, 6 - with b=2pi)

        Notes
        -----
        The SHAPE of any of the below args indicates the *ordering* of the
        raveled array, with last axis moving first.

        Args
        ----
        ndata (int) :
            Number of different data sets to be transformed (each is independent)
        n_nu (int) :
            Number of real-space cells (eg. frequencies, in terms of visibilities)
        n_eta (int) :
            Number of fourier-space cells to transform to, should be ~1/2 n_nu
        conv_ext (double) :
            Convergence extent. The number of Morlet kernel sigma "widths" to
            actually perform integration for. Should be ~5 or more.
        fourier_b (double) :
            The Fourier convention, i.e. the Fourier kernel is e^{-bi nu*eta}.
        data (double complex, SHAPE=[n_nu, ndata]) :
            The input (complex) data.
        nu (double, SHAPE=[n_nu]):
            Real-space co-ordinates (i.e. frequencies, in terms of visibilities)
        nthreads (int) :
            Number of threads to use in OMP.
        eta (double, SHAPE=[n_eta]):
            Fourier-space co-ordinates (dual of nu).

        Returns
        -------
        out (double complex, SHAPE=[n_eta, n_nu, ndata]):
            The resulting Morlet transform.
    */

    unsigned int ix, jnuc,jeta, jnu,  jidx, jmin, jmax;
    double exponent;
    double complex xx;

    int thisn;

    double sqrt2 = sqrt(2.0);
    unsigned int out_idx = 0;
    unsigned int data_idx = 0;

    double sqrt2dnu = sqrt2*(nu[1] - nu[0]);

    omp_set_num_threads(nthreads);

    #pragma omp parallel for private(thisn, jidx, out_idx, jnuc, jmin, jmax, data_idx, exponent, xx, jnu, ix)
    for (jeta=0; jeta<n_eta;jeta++){ // Loop through eta

        thisn = ceil(conv_ext/(eta[jeta]*sqrt2dnu));

        // We do this to be able to multi-thread
        jidx = jeta * n_nu * ndata;
        out_idx = 0;

        LOG_DEBUG("jeta=%d, jidx=%d, thisn=%d", jeta, jidx, thisn);

        for (jnuc=0;jnuc<n_nu;jnuc++){ // Loop through nu_centre
            jmin = max(0, jnuc-thisn);
            jmax = min(jnuc+thisn, n_nu);

            data_idx = jmin*ndata;

            LOG_SUPER_DEBUG("jnuc=%d, jmin=%d, jmax=%d", jnuc, jmin, jmax);

            for (jnu=jmin; jnu<jmax; jnu++){ // Loop through nu (i.e. do the FT)
                exponent = eta[jeta]*(nu[jnu] - nu[jnuc]);
                xx = cexp(-exponent*(exponent/2 + fourier_b*I));

                for (ix=0;ix<ndata;ix++){  // Loop through different data
//                    if(jidx + out_idx >= n_eta*n_nu*ndata){
//                        printf("Out of bounds on: jeta=%d, jnuc=%d, jnu=%d, ix=%d, jidx=%d, out_idx=%d\n", jeta, jnuc, jnu, ix, jidx, out_idx);
//                    }

                    out[jidx + out_idx] += data[data_idx]*xx;

                    if(jeta==(n_eta-1) && jnuc==(n_nu-1) && ix==(ndata-1))
                      LOG_ULTRA_DEBUG("\t\tjnu=%d ix=%d indx=%d jidx=%d, out_idx=%d, data=%g + %gi xx=%g out=%g + %gi", jnu, ix, jidx+out_idx, jidx, out_idx, creal(data[data_idx]), cimag(data[data_idx]), xx, creal(out[jidx + out_idx]), cimag(out[jidx + out_idx]));

                    data_idx++;
                    out_idx++;
                }
                out_idx -= ndata; // out_idx should not contain jnu, so reset it.
            }
            out_idx += ndata;
        }
    }
}

void BlackmanHarris_cmorlet(unsigned int ndata, unsigned int n_nu, unsigned int n_eta,
             double conv_ext, double *BlackmanHarrisFilter,
             double complex *data, double *nu, double *eta, int nthreads,
             double complex *out){
    /*
        Discrete Morlet Wavelet transform but replacing Gaussian filter with BlackmanHarris.
        BlackmanHarris form taken from scipy.signal.windown.blackmanharris (copied below)
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.blackmanharris.html#scipy.signal.windows.blackmanharris

        Otherwise as above.
        =================================
    */

    unsigned int ix, jnuc,jeta, jnu,  jidx, jmin, jmax;
    double exponent;
    double complex xx;

    int thisn;

    double sqrt2 = sqrt(2.0);
    unsigned int out_idx = 0;
    unsigned int data_idx = 0;

    double sqrt2dnu = sqrt2*(nu[1] - nu[0]);

    omp_set_num_threads(nthreads);

    #pragma omp parallel for private(thisn, jidx, out_idx, jnuc, jmin, jmax, data_idx, exponent, xx, jnu, ix)
    for (jeta=0; jeta<n_eta;jeta++){ // Loop through eta

        thisn = ceil(conv_ext/(eta[jeta]*sqrt2dnu));

        // We do this to be able to multi-thread
        jidx = jeta * n_nu * ndata;
        out_idx = 0;

        LOG_DEBUG("jeta=%d, jidx=%d, thisn=%d", jeta, jidx, thisn);

        for (jnuc=0;jnuc<n_nu;jnuc++){ // Loop through nu_centre
            jmin = max(0, jnuc-thisn);
            jmax = min(jnuc+thisn, n_nu);


            data_idx = jmin*ndata;

            LOG_SUPER_DEBUG("jnuc=%d, jmin=%d, jmax=%d", jnuc, jmin, jmax);

            for (jnu=jmin; jnu<jmax; jnu++){ // Loop through nu (i.e. do the FT)
                exponent = eta[jeta]*(nu[jnu] - nu[jnuc]);
                //xx = cexp(-exponent*(exponent/2 + fourier_b*I)); // --Filter change Fourier to blackmanharris.
                //general_cosine(M, [0.35875, 0.48829, 0.14128, 0.01168], sym)
                xx = cexp(-exponent*exponent/2 - BlackmanHarrisFilter[jnuc]);

                for (ix=0;ix<ndata;ix++){  // Loop through different data
//                    if(jidx + out_idx >= n_eta*n_nu*ndata){
//                        printf("Out of bounds on: jeta=%d, jnuc=%d, jnu=%d, ix=%d, jidx=%d, out_idx=%d\n", jeta, jnuc, jnu, ix, jidx, out_idx);
//                    }


                    out[jidx + out_idx] += data[data_idx]*xx;

                    if(jeta==(n_eta-1) && jnuc==(n_nu-1) && ix==(ndata-1))
                      LOG_ULTRA_DEBUG("\t\tjnu=%d ix=%d indx=%d jidx=%d, out_idx=%d, data=%g + %gi xx=%g out=%g + %gi", jnu, ix, jidx+out_idx, jidx, out_idx, creal(data[data_idx]), cimag(data[data_idx]), xx, creal(out[jidx + out_idx]), cimag(out[jidx + out_idx]));

                    data_idx++;
                    out_idx++;
                }
                out_idx -= ndata; // out_idx should not contain jnu, so reset it.
            }
            out_idx += ndata;
        }
    }
}


/* def blackmanharris(M, sym=True):
    """Return a minimum 4-term Blackman-Harris window.
    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty
        array is returned.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.
    Returns
    -------
    w : ndarray
        The window, with the maximum value normalized to 1 (though the value 1
        does not appear if `M` is even and `sym` is True).
    Examples
    --------
    Plot the window and its frequency response:
    >>> from scipy import signal
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt
    >>> window = signal.windows.blackmanharris(51)
    >>> plt.plot(window)
    >>> plt.title("Blackman-Harris window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    >>> plt.figure()
    >>> A = fft(window, 2048) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    >>> plt.plot(freq, response)
    >>> plt.axis([-0.5, 0.5, -120, 0])
    >>> plt.title("Frequency response of the Blackman-Harris window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    """
    return general_cosine(M, [0.35875, 0.48829, 0.14128, 0.01168], sym)

def general_cosine(M, a, sym=True):
    r"""
    Generic weighted sum of cosine terms window
    Parameters
    ----------
    M : int
        Number of points in the output window
    a : array_like
        Sequence of weighting coefficients. This uses the convention of being
        centered on the origin, so these will typically all be positive
        numbers, not alternating sign.
    sym : bool, optional
        When True (default), generates a symmetric window, for use in filter
        design.
        When False, generates a periodic window, for use in spectral analysis.
    References
    ----------
    .. [1] A. Nuttall, "Some windows with very good sidelobe behavior," IEEE
           Transactions on Acoustics, Speech, and Signal Processing, vol. 29,
           no. 1, pp. 84-91, Feb 1981. :doi:`10.1109/TASSP.1981.1163506`.
    .. [2] Heinzel G. et al., "Spectrum and spectral density estimation by the
           Discrete Fourier transform (DFT), including a comprehensive list of
           window functions and some new flat-top windows", February 15, 2002
           https://holometer.fnal.gov/GH_FFT.pdf
    Examples
    --------
    Heinzel describes a flat-top window named "HFT90D" with formula: [2]_
    .. math::  w_j = 1 - 1.942604 \cos(z) + 1.340318 \cos(2z)
               - 0.440811 \cos(3z) + 0.043097 \cos(4z)
    where
    .. math::  z = \frac{2 \pi j}{N}, j = 0...N - 1
    Since this uses the convention of starting at the origin, to reproduce the
    window, we need to convert every other coefficient to a positive number:
    >>> HFT90D = [1, 1.942604, 1.340318, 0.440811, 0.043097]
    The paper states that the highest sidelobe is at -90.2 dB.  Reproduce
    Figure 42 by plotting the window and its frequency response, and confirm
    the sidelobe level in red:
    >>> from scipy.signal.windows import general_cosine
    >>> from scipy.fft import fft, fftshift
    >>> import matplotlib.pyplot as plt
    >>> window = general_cosine(1000, HFT90D, sym=False)
    >>> plt.plot(window)
    >>> plt.title("HFT90D window")
    >>> plt.ylabel("Amplitude")
    >>> plt.xlabel("Sample")
    >>> plt.figure()
    >>> A = fft(window, 10000) / (len(window)/2.0)
    >>> freq = np.linspace(-0.5, 0.5, len(A))
    >>> response = np.abs(fftshift(A / abs(A).max()))
    >>> response = 20 * np.log10(np.maximum(response, 1e-10))
    >>> plt.plot(freq, response)
    >>> plt.axis([-50/1000, 50/1000, -140, 0])
    >>> plt.title("Frequency response of the HFT90D window")
    >>> plt.ylabel("Normalized magnitude [dB]")
    >>> plt.xlabel("Normalized frequency [cycles per sample]")
    >>> plt.axhline(-90.2, color='red')
    >>> plt.show()
    """
    if _len_guards(M):
        return np.ones(M)
    M, needs_trunc = _extend(M, sym)

    fac = np.linspace(-np.pi, np.pi, M)
    w = np.zeros(M)
    for k in range(len(a)):
        w += a[k] * np.cos(k * fac)

    return _truncate(w, needs_trunc)
    */

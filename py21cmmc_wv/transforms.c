#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <omp.h>

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

void cmorlet(int ndata, int n_nu, int n_eta, double conv_ext, double fourier_b,
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

    unsigned int ix, jnuc,jeta, jnu, thisn, jidx;
    double exponent, mag, dt;
    double complex xx;

    double sqrt2 = sqrt(2.0);
    unsigned int out_idx = 0;
    unsigned int data_idx = 0;

    double sqrt2dnu = sqrt2*(nu[1] - nu[0]);

    #pragma omp parallel for private(thisn, jidx, out_idx, jnuc, jmin, jmax, data_idx, exponent, xx, jnu, ix)
    for (jeta=0; jeta<n_eta;jeta++){ // Loop through eta

        thisn = ceil(conv_ext/(eta[jeta]*sqrt2dnu));

        // We do this to be able to multi-thread
        jidx = jeta * n_nu * ndata;
        out_idx = 0;

        for (jnuc=0;jnuc<n_nu;jnuc++){ // Loop through nu_centre
            jmin = fmax(0, jnuc-thisn);
            jmax = fmin(jnuc+thisn, n_nu);

            data_idx = 0;
            for (jnu=jmin; jnu<jmax; jnu++){ // Loop through nu (i.e. do the FT)
                exponent = eta[jeta]*(nu[jnu] - nu[jnuc]);
                xx = cexp(-exponent*(exponent/2 + fourier_b*I));
                
                for (ix=0;ix<ndata;ix++){  // Loop through different data
                    out[jidx + out_idx] += data[data_idx]*xx;
                    data_idx++;
                    out_idx++;
                }
                out_idx -= ndata; // out_idx should not contain jnu, so reset it.
            }
        }
    }
}
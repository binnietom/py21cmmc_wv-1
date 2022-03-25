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
                xx = cexp(-exponent*(exponent/2 + fourier_b*I)); ////THIS BIT IS THE ENVELOPE
                //xx = cexp(-BlackmanHarris +

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

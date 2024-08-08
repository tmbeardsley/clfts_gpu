// #######################################################################################
// Provides the public method: void fwd(...), which takes the propagators of the previous
// monomer as input and returns the propagators of the next monomer as output
// #######################################################################################

#pragma once
#include <cuda.h>
#include <cufft.h>
#include "GPUerror.h"
#include <math.h>
#include <thrust/device_vector.h>




class step {
    // Step-specific variables
    int TpB_;                                                                   // GPU threads per block (default: 512)

    thrust::device_vector<thrust::complex<double>> g_gpu_;                      // Bond potential Boltzmann weight, Fourier transformed and /M_ on the GPU
    cufftHandle qr_to_qk_;                                                      // cufft plan to transform q1[r] and q2[r] to k-space
    const int cuFFTFORWARD_ = -1;
    const int cuFFTINVERSE_ = 1;

    // Simulation constants derived from the input file (see lfts_params.h for details)
    int NA_;
    int NB_;
    int* m_;
    int M_;

    public:
        // Constructor
        step(int NA, int NB, int* m, double* L, int M, int TpB = 512) {
            TpB_ = TpB;
            NA_ = NA;
            NB_ = NB;

            m_ = new int[3];
            for (int i = 0; i < 3; i++) m_[i] = m[i];

            M_ = M;

            // Allocate memory for g_gpu.
            // g_gpu_ contains two copies of g[] so that q1[k] and q2[k] can be multiplied on the GPU at the same time
            g_gpu_.resize(2*M);

            // Calculate the lookup table for g_gpu_ (copied to gpu in function for box move to be added later)
            update_g_lookup(L);

            // Configure cufft plans. cufftPlanMany used for batched processing
            GPU_ERR(cufftPlanMany(&qr_to_qk_, 3, m_, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 2));
        }







        // Takes the propagators of the previous monomer (index, i) as input and returns the propagators 
        // of the next monomer as output.
        // Works with real- or k-space representations of the propagators, as indicated by bool KSPACE.
        void fwd(   thrust::device_vector<thrust::complex<double>>::iterator& q_in,
                    thrust::device_vector<thrust::complex<double>>::iterator& q_out,
                    thrust::device_vector<thrust::complex<double>>::iterator& h_gpu,
                    int i, bool KSPACE = false)
        {
            // hX initially used as workspace memory (later represents hA(r) or hB(r))
            thrust::device_vector<thrust::complex<double>>::iterator hX = q_in;

            // forward FFT q_i(r) and q^\dagger_{N+1-i}(r)
            cufftDoubleComplex *V1 = (cufftDoubleComplex*)thrust::raw_pointer_cast(&(q_in[0]));
            cufftDoubleComplex *V2 = (cufftDoubleComplex*)thrust::raw_pointer_cast(&(q_out[0]));

            // Perform a fourier transform if calculating real-space propagators
            if (!KSPACE) {
                GPU_ERR(cufftExecZ2Z(qr_to_qk_, V1, V2, cuFFTFORWARD_));
                hX = q_out;
            }

            // multiply by g[k]
            thrust::transform(hX, hX + 2*M_, g_gpu_.begin(), q_out, thrust::multiplies<thrust::complex<double>>());

            // inverse FFT to obtain convolution integrals
            GPU_ERR(cufftExecZ2Z(qr_to_qk_, V2, V2, cuFFTINVERSE_));

            // multiple by hA[r] or hB[r] to obtain q_{i+1}(r)
            hX = (i < NA_) ? h_gpu : h_gpu + M_;
            thrust::transform(q_out, q_out + M_, hX, q_out, thrust::multiplies<thrust::complex<double>>());

            // multiple by hA[r] or hB[r] to obtain q^dagger_{N+1-i}(r)
            hX = (i < NB_) ? h_gpu + M_ : h_gpu;
            thrust::transform(q_out + M_, q_out + 2 * M_, hX, q_out + M_, thrust::multiplies<thrust::complex<double>>());

            // perform additional Fourier transform if calculating k-space propagators
            if (KSPACE) GPU_ERR(cufftExecZ2Z(qr_to_qk_, V2, V2, cuFFTFORWARD_));

        }


        // Provide access to g_gpu for the get_Q_derivatives() function in diblock class()
        thrust::device_vector<thrust::complex<double>>::iterator g_gpu()
        {
            return g_gpu_.begin();
        }


        // Destructor
        ~step() {
            GPU_ERR(cufftDestroy(qr_to_qk_));
            delete[] m_;
        }


    private:
        // Calculate the Boltzmann weight of the bond potential in k-space, _g[k]
        void update_g_lookup(double* L) {
            int K0, K1, K2, k, N = NA_ + NB_;
            double K, kx_sq, ky_sq, kz_sq;
            thrust::device_vector<thrust::complex<double>> g(M_);

            for (int k0 = -(m_[0] - 1) / 2; k0 <= m_[0] / 2; k0++) {
                K0 = (k0 < 0) ? (k0 + m_[0]) : k0;
                kx_sq = k0 * k0 / (L[0] * L[0]);

                for (int k1 = -(m_[1] - 1) / 2; k1 <= m_[1] / 2; k1++) {
                    K1 = (k1 < 0) ? (k1 + m_[1]) : k1;
                    ky_sq = k1 * k1 / (L[1] * L[1]);

                    for (int k2 = -(m_[2] - 1) / 2; k2 <= m_[2] / 2; k2++) {
                        K2 = (k2 < 0) ? (k2 + m_[2]) : k2;
                        kz_sq = k2 * k2 / (L[2] * L[2]);
                        k = K2 + m_[2]*(K1+m_[1]*K0);
                        K = 2 * M_PI * pow(kx_sq + ky_sq + kz_sq, 0.5);
                        g[k] = exp(-K * K / (6.0 * N)) / M_;
                    }
                }
            }

            thrust::copy(g.begin(), g.end(), g_gpu_.begin());
            thrust::copy(g.begin(), g.end(), g_gpu_.begin() + M_);
        }

};





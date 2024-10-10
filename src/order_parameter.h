#pragma once
#include <cuda.h>
#include <cufft.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <math.h>
#include "file_IO.h"

// w(k) = get<0>(t)
// w(-k) = get<1>(t)
// f(k) = get<2>(t)
// Calculate norm(w(k)*w(-k))*f(k) on the gpu.
struct Psi4_calc
{
    Psi4_calc() {}
	__device__ __host__
	double operator()(thrust::tuple<thrust::complex<double>, thrust::complex<double>, double> t) {
        return thrust::norm(thrust::get<0>(t)*thrust::get<1>(t))*thrust::get<2>(t);
	}
};



class order_parameter {

    thrust::device_vector<thrust::complex<double>> wk_gpu_;             // Composition field in reciprocal space, w-(k), on the GPU
    thrust::device_vector<double> fk_gpu_;                              // Weighting function to reduce the contribution of large wavevectors in the order parameter
    thrust::device_vector<int> *minusK_gpu_;                            // Given the 1D index, k, then minusK[k] is the 1D index pointing to the negative wavevector in array w[k]
    cufftHandle wr_to_wk_;						// Cufft handle for setting up a Fourier transform on the GPU
    int M_;								// Total number of meshpoints
    double kc_;								// Wavevector cutoff in f(k)
    const int cuFFTFORWARD = -1;					// Direction of Fourier transform (real space to k-space)


    public:
        // Constructor
        order_parameter(int *m, double *L, int M, thrust::device_vector<int> *minusK_gpu, double kc = 6.02) {
            M_ = M;
            kc_ = kc;

	    // Allocate memory for wk_gpu_ and fk_gpu_
            wk_gpu_.resize(M);
            fk_gpu_.resize(M);

	    // Create a cufft plan to transform fields between real and reciprocal space
            GPU_ERR(cufftPlan3d(&wr_to_wk_, m[0], m[1], m[2], CUFFT_Z2Z));

            // Create lookup tables for fk[] and minusK[] and upload to GPU
            thrust::host_vector<double> fk(M);
            calc_fk(&(fk[0]), kc, m, L);
            fk_gpu_ = fk;

            // Keep a pointer to the minusK_gpu lookup table
            minusK_gpu_ = minusK_gpu;
        }


        // Destructor
        ~order_parameter() {
            GPU_ERR(cufftDestroy(wr_to_wk_));
        }


        // Calculate the order parameter with fixed parameter, ell = 4 (ensures Psi is real)
        double get_Psi4(thrust::device_vector<thrust::complex<double>> &w_gpu, bool isWkInput = false)
        {
            // Start by assuming the function has been passed w(k)
            thrust::device_vector<thrust::complex<double>>::iterator wk_itr = w_gpu.begin();

            if (!isWkInput) {
                // Fourier transform w-(r) to get w-(k)
                GPU_ERR(cufftExecZ2Z(wr_to_wk_, 
                                    (cufftDoubleComplex*)thrust::raw_pointer_cast(&(w_gpu[0])), 
                                    (cufftDoubleComplex*)thrust::raw_pointer_cast(&(wk_gpu_[0])),
                                    cuFFTFORWARD));
                wk_itr = wk_gpu_.begin();
            }

            // Perform a thrust transform reduction on the gpu and calculate Psi
            auto minusK_iter = thrust::make_permutation_iterator(wk_itr, (*minusK_gpu_).begin());
            auto z = thrust::make_zip_iterator(thrust::make_tuple(wk_itr, minusK_iter, fk_gpu_.begin()));
            double Psi = thrust::transform_reduce(z, z + M_, Psi4_calc(), 0.0, thrust::plus<double>());
            return pow(Psi/(1.0*M_*M_), 0.25);  // *1.0 avoids integer overflow
        }


    private:
        // Calculate the function, f(k), to screen out the effects of large wave-vectors
        void calc_fk(double *fk, double kc, int *m, double *L) {
            int K0, K1, K2, k;
            double kx_sq, ky_sq, kz_sq, K;

            for (int k0=-(m[0]-1)/2; k0<=m[0]/2; k0++) {
                K0 = (k0<0)?(k0+m[0]):k0;
                kx_sq = k0*k0/(L[0]*L[0]);

                for (int k1=-(m[1]-1)/2; k1<=m[1]/2; k1++) {
                    K1 = (k1<0)?(k1+m[1]):k1;
                    ky_sq = k1*k1/(L[1]*L[1]);

                    for (int k2=-(m[2]-1)/2; k2<=m[2]/2; k2++) {
                        K2 = (k2<0)?(k2+m[2]):k2;
                        kz_sq = k2*k2/(L[2]*L[2]);

                        k = K2 + m[2]*(K1+m[1]*K0);
                        K = 2*M_PI*pow(kx_sq+ky_sq+kz_sq,0.5);

                        fk[k] = 1.0/(1.0 + exp(12.0*(K-kc)/kc));
                    }
                }
            }
        }

};

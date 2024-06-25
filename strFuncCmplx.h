// #####################################################################################
// Provides the public methods: void sample(...) and void save(...), 
// which take samples of the structure funtion, S(k), and save the spherically-averaged 
// S(k) to file.
// S(k) should only be calculated in simulations keeping L[] and XbN constant.
// #####################################################################################

#pragma once
#include <cuda.h>
#include <cufft.h>
#include "GPUerror.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <math.h>
#include <complex>
#include <fstream>
#include "sorts.h"


struct wk_sq_functor
{
    wk_sq_functor() {}
    template <typename Tuple>
    __host__ __device__
        void operator()(Tuple t) { // S[k]=get<0>(t), wk[k]=get<1>(t), wk[minusK[k]]=get<2>(t)
        thrust::get<0>(t) += (thrust::get<1>(t) * thrust::get<2>(t)).real();
    }
};


class strFuncCmplx {              
    double dK_;                     // Maximum allowed difference used to define like wave vectors for spherical averaging
    double coeff_;                  // A constant used in saving the structure function
    int nsamples_;                  // Number of structure function samples taken
    cufftHandle wr_to_wk_;          // cufft plan transforming w-(r) to w-(k)
    const int cuFFTFORWARD_ = -1;   // Direction of Fourier Transform (used in cufft calls)

    double *K_;                     // Modulus of wavevector k
    int *P_;                        // Map transforming K_[] into ascending order
    
    thrust::device_vector<thrust::complex<double>> wk_gpu_;     // w-(k) on the GPU
    thrust::device_vector<double> S_gpu_;                       // Collects sum of |wk_[k]| resulting from calls to: sample(double *w_gpu)
    thrust::device_vector<int> minusK_gpu_;                     // Given the 1D index, k, then minusK[k] is the 1D index pointing to the negative wavevector in array K[k]
    thrust::permutation_iterator<   thrust::device_vector<thrust::complex<double>>::iterator, 
                                    thrust::device_vector<int>::iterator
                                >   minusK_iter_;               // Permutation iterator for multiplying wk[k] by wk[minusK[k]] on the GPU

    // Simulation constants derived from the input file (see lfts_params.h for details)
    double chi_b_;
    int M_;

    public:
        // Constructor
        strFuncCmplx(int *m, double *L, int M, double CV, double chi_b, double dK = 1E-5) {
            dK_ = dK;
            M_ = M;
            chi_b_ = chi_b;
            nsamples_ = 0;
            coeff_ = CV/(chi_b*chi_b*M*M);
            K_ = new double[M];
            P_ = new int[M];

            // Allocate memory for w-(-k) on the GPU
            minusK_gpu_.resize(M);

            // Allocate memory for w-(k) on the GPU
            wk_gpu_.resize(M);

            // Allocate memory for S(k) on the GPU and zero all elements
            S_gpu_.resize(M);
            thrust::fill(S_gpu_.begin(), S_gpu_.end(), 0.0);

            // Create a cufft plan for the Fourier transform on the GPU
            GPU_ERR(cufftPlan3d(&wr_to_wk_, m[0], m[1], m[2], CUFFT_Z2Z));

            // Populate the wavevector arrays, K_, minusK_gpu
            thrust::host_vector<int> minusK(M);
            //calcK(K_, (int*)thrust::raw_pointer_cast(&(minusK[0])), m, L);
            calcK(K_, &(minusK[0]), m, L);
            minusK_gpu_ = minusK;

            // Create Permutation iterator for use in wk_sq_functor() functor
            minusK_iter_ = thrust::make_permutation_iterator(wk_gpu_.begin(), minusK_gpu_.begin());

            // Populate the map, P_, which puts the wavevector moduli, K_, into ascending order
            for (int k=0; k<M; k++) P_[k] = k;
            sorts::quicksortMap(K_, P_, 0, M-1);
        }






        // Sample norm(w-(k)) 
        void sample(thrust::device_vector<thrust::complex<double>> w_gpu) {
            // Transform w-(r) to k-space to get w-(k)
            GPU_ERR(cufftExecZ2Z(wr_to_wk_, (cufftDoubleComplex*)thrust::raw_pointer_cast(&(w_gpu[0])), (cufftDoubleComplex*)thrust::raw_pointer_cast(&(wk_gpu_[0])), cuFFTFORWARD_));

            // Make a zip iterator for passing to wk_sq_functor() 
            auto z = thrust::make_zip_iterator(thrust::make_tuple(S_gpu_.begin(), wk_gpu_.begin(), minusK_iter_));
            
            // Sample the norm of w-(k) for each wavevector and add to its sum
            thrust::for_each(z, z + M_, wk_sq_functor());

            // Increment the number of samples
            nsamples_++;
        }






        // Output the spherically-averaged structure function to file
        void save(std::string fileName, int dp=8) {
            double S_sum = 0.0;
            int k, n_same = 0;
            std::ofstream out_stream;

            out_stream.open(fileName);
            out_stream.precision(dp);
            out_stream.setf(std::ios::fixed, std::ios::floatfield);

            // Copy S_gpu to the host
            thrust::host_vector<double> S(M_);
            S = S_gpu_;

            // Spherical average of S(k)
            for (k=0; k<M_; k++) {
                // Take into account vector weighting from the FFT and sum S for repeated K-vectors
                S_sum += (coeff_/nsamples_)*S[P_[k]] - 0.5/chi_b_;
                n_same ++;

                // Output value for current K-vector when difference in K exceeds tolerence dK_
                if ( (k==M_-1) || (fabs(K_[P_[k+1]]-K_[P_[k]]) > dK_) ) {
                    out_stream << K_[P_[k]] << "\t" << S_sum/n_same << std::endl;

                    // Reset summations for next K-vector
                    S_sum = 0.0;
                    n_same = 0;
                }
            } 
            out_stream.close();
        }

        // Destructor
        ~strFuncCmplx() {
            delete[] K_;
            delete[] P_;
            GPU_ERR(cufftDestroy(wr_to_wk_));
        }




    private:
        // Calculate the wavevector moduli and store in K[]
        void calcK(double *K, int *minusK, int *m, double *L) {
            int K0, K1, K2, k;
            int mK0, mK1, mK2;
            double kx_sq, ky_sq, kz_sq;

            for (int k0=-(m[0]-1)/2; k0<=m[0]/2; k0++) {
                K0 = (k0<0)?(k0+m[0]):k0;
                mK0 = (k0>0)?(-k0+m[0]):-k0;
                kx_sq = k0*k0/(L[0]*L[0]);

                for (int k1=-(m[1]-1)/2; k1<=m[1]/2; k1++) {
                    K1 = (k1<0)?(k1+m[1]):k1;
                    mK1 = (k1>0)?(-k1+m[1]):-k1;
                    ky_sq = k1*k1/(L[1]*L[1]);

                    for (int k2=-(m[2]-1)/2; k2<=m[2]/2; k2++) {
                        K2 = (k2<0)?(k2+m[2]):k2;
                        mK2 = (k2>0)?(-k2+m[2]):-k2;
                        kz_sq = k2*k2/(L[2]*L[2]);

                        k = K2+m[2]*(K1+m[1]*K0);
                        minusK[k] = mK2+m[2]*(mK1+m[1]*mK0);
                        K[k] = 2*M_PI*pow(kx_sq+ky_sq+kz_sq,0.5); 
                    }
                }
            }
        }

};
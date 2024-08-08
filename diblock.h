// ###########################################################################################
// Provides public method: calc_concs(thrust::device_vector<thrust::complex<double>> &w_gpu), 
// to calculate concentrations
// ###########################################################################################

#pragma once
#include <cuda.h>
#include <cufft.h>
#include <vector>
#include <math.h>
//#include "GPUkernels.h"
#include "GPUerror.h"
#include "step.h"
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>

struct h_functor {
    const int N_;
    int sign_;
    h_functor(int N, bool hA=true) : N_(N) {
        sign_ = hA ? +1 : -1;
    }
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t) {
        // t = { h, wm_gpu, wp_gpu }
        thrust::get<0>(t) = exp(-(thrust::get<2>(t) + sign_*thrust::get<1>(t)) / N_);
    }
};

// Calculate phi- and phi+
// Caution: hA and hB get overwritten
struct normalise_functorer {
    const thrust::complex<double> Q_;
    const int N_;
    normalise_functorer(thrust::complex<double> Q, int N) : Q_(Q), N_(N) {}
    template <typename Tuple>
    __host__ __device__
        void operator()(Tuple t) { 
        // t = { phiA, phiB, hA, hB }
        thrust::get<2>(t) = thrust::get<0>(t) / (N_ * Q_ * thrust::get<2>(t));
        thrust::get<3>(t) = thrust::get<1>(t) / (N_ * Q_ * thrust::get<3>(t));
        thrust::get<0>(t) = thrust::get<2>(t) - thrust::get<3>(t);
        thrust::get<1>(t) = thrust::get<2>(t) + thrust::get<3>(t);
    }
};

// Calculates A += B*C
struct sum_prod_functor {
    sum_prod_functor() {}
    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t) { 
        // t = { A, B, C }
        thrust::get<0>(t) += thrust::get<1>(t) * thrust::get<2>(t);
    }
};


struct dQdL_functor
{
    __host__ __device__
    thrust::complex<double> operator()(thrust::tuple<double, thrust::complex<double>, thrust::complex<double>> t) {
        thrust::complex<double> result = thrust::get<0>(t)*thrust::get<1>(t)* thrust::get<2>(t);
        return result;
    }
};





class diblockClass {

    // Diblock-specific variables
    int TpB_;
    step *Step_;                                                                            // Step object to get propagators for the next monomer

    thrust::device_vector<thrust::complex<double>> qr_gpu_;                                 // Pointer to GPU memory for propagators: q_{i}(r) and q^_{N+1-i}(r) are contigious in memory

    thrust::device_vector<thrust::complex<double>> h_gpu_;                                  // GPU memory for hA(r) and hB(r)
    thrust::device_vector<thrust::complex<double>>::iterator hA_gpu_, hB_gpu_;

    thrust::host_vector<thrust::device_vector<thrust::complex<double>>::iterator> q1_;      // Array of pointers to q_{j=i}(r), where j is the monomer index and i is array index
    thrust::host_vector<thrust::device_vector<thrust::complex<double>>::iterator> q2_;      // Array of pointers to q^_{j=N+1-i}(r), where j is the monomer index and i is array index
    thrust::device_vector<int> *minusK_gpu_;                                                // Given the 1D index, k, then minusK[k] is the 1D index pointing to the negative wavevector in array K[k]
    thrust::device_vector<double> c_dQdL_gpu_;                                              // Constants for use in calculation of dQ/dL
    thrust::device_vector<thrust::complex<double>> SUM_gpu_;                                //
    cufftHandle FFT_plan_;
    const int cuFFTFORWARD = -1;

    thrust::complex<double> Q_;                                                             // Partition function after the last call to calc_concs()

    // Simulation constants derived from the input file (see lfts_params.h for details)
    int M_;
    int NA_;
    int N_;
    std::vector<double> L_;



    public:
        // Constructor
        diblockClass(int NA, int NB, int* m, double* L, int M, thrust::device_vector<int> *minusK_gpu)
        {
            M_ = M;
            NA_ = NA;
            N_ = NA+NB;
            L_.resize(3);
            for (int i=0; i<3; i++) L_[i] = L[i];

            Q_ = 0.0;

            // Allocate gpu memory for h_gpu_ and qr_gpu_
            h_gpu_.resize(2*M);
            hA_gpu_ = h_gpu_.begin();
            hB_gpu_ = hA_gpu_ + M;

            qr_gpu_.resize(2*(N_+1)*M);

            // Allocate arrays of iterators for q_{j=1...N}(r) and q^_{j=1...N}(r)
            q1_.resize(N_+1);
            q2_.resize(N_+1);

            // Assign iterators such that q_{1}(r) and q_{N}(r) are in contigious memory,
            // as are q_{2}(r) and q_{N-1}(r), q_{3}(r) and q_{N-2}(r)... etc. (required for cufftPlanMany())
            for (int i=1; i<=N_; i++) {
                q1_[i] = qr_gpu_.begin() + 2*i*M;
                q2_[N_+1-i] = qr_gpu_.begin() + (2*i+1)*M;
            }

            // Allocate memory for w-(-k) and c_dQdL on the GPU
            //minusK_gpu_.resize(M);
            c_dQdL_gpu_.resize(3*M);
            SUM_gpu_.resize(M);

            //thrust::host_vector<int> minusK(M);
            thrust::host_vector<double> c_dQdL(3*M);
            calcK(&(c_dQdL[0]), m, &(L_[0]));
            //minusK_gpu_ = minusK;
            c_dQdL_gpu_ = c_dQdL;

            // Keep a pointer to the minusK_gpu lookup table
            minusK_gpu_ = minusK_gpu;

            // Set up a cuda plan
            GPU_ERR(cufftPlanMany(&FFT_plan_,3,m,NULL,1,0,NULL,1,0, CUFFT_Z2Z,2));

            // New step object containing methods to get next monomer's propagators
            Step_ = new step(NA, NB, m, L, M);
        }





        // Returns the partition function from the most recent call to calc_concs()
        thrust::complex<double> Q() { return Q_; }






        // Calculates phi-(r) and phi+(r): w+2*M -> phi-(0), w+3*M -> phi+(0).
        // Returns ln(Q)
        thrust::complex<double> calc_concs(thrust::device_vector<thrust::complex<double>> &w_gpu)
        {
            int i;

            thrust::device_vector<thrust::complex<double>>::iterator phim_gpu = w_gpu.begin() + 2*M_;
            thrust::device_vector<thrust::complex<double>>::iterator phip_gpu = w_gpu.begin() + 3*M_;

            // Calculate hA[r] and hB[r] on the GPU
            auto z = thrust::make_zip_iterator(thrust::make_tuple(hA_gpu_, w_gpu.begin(), w_gpu.begin()+M_));
            thrust::for_each(z, z + M_, h_functor(N_, true));
            z = thrust::make_zip_iterator(thrust::make_tuple(hB_gpu_, w_gpu.begin(), w_gpu.begin()+M_));
            thrust::for_each(z, z + M_, h_functor(N_, false));


            // Set initial conditions: q[1][r]=hA[r] and q^[N][r]=hB[r] for all r
            thrust::copy(h_gpu_.begin(), h_gpu_.end(), q1_[1]);

            // Step the propagators q1 and q2 for each subsequent monomer (note q[i],q^[N+1-i]... contigious in memory)
            for (i=1; i<N_; i++) Step_->fwd(q1_[i], q1_[i+1], hA_gpu_, i);

            // Calculate single-chain partition function using a Thrust reduction sum
            Q_ = thrust::reduce(q1_[N_], q1_[N_]+M_) / M_;

            // zero the concentrations
            thrust::fill(phim_gpu, phim_gpu + 2*M_, 0.0);

            // Calculate concentrations
            for (i = 1; i <= NA_; i++) {
                z = thrust::make_zip_iterator(thrust::make_tuple(phim_gpu, q1_[i], q2_[i]));
                thrust::for_each(z, z + M_, sum_prod_functor());
            }
            for (i = NA_+1; i <= N_; i++) {
                z = thrust::make_zip_iterator(thrust::make_tuple(phip_gpu, q1_[i], q2_[i]));
                thrust::for_each(z, z + M_, sum_prod_functor());
            }

            // normalise the concentrations
            // WARNING: Contents of hA_gpu_ and hB_gpu_ get overwritten to save memory
            auto zb = thrust::make_zip_iterator(thrust::make_tuple(phim_gpu, phip_gpu, hA_gpu_, hB_gpu_));
            thrust::for_each(zb, zb+M_, normalise_functorer(Q_, N_));

            return thrust::log(Q_);
        }





        // Get dQ/dL components
        // Returns Q
        thrust::complex<double> get_Q_derivatives(thrust::device_vector<thrust::complex<double>> &w_gpu, thrust::host_vector<thrust::complex<double>> &dQdL)
        {
            int i;
            thrust::complex<double> cmplx_zero = {0.0, 0.0};

            // Calculate hA[r] and hB[r] on the GPU
            auto z = thrust::make_zip_iterator(thrust::make_tuple(hA_gpu_, w_gpu.begin(), w_gpu.begin()+M_));
            thrust::for_each(z, z+M_, h_functor(N_, true));
            z = thrust::make_zip_iterator(thrust::make_tuple(hB_gpu_, w_gpu.begin(), w_gpu.begin()+M_));
            thrust::for_each(z, z+M_, h_functor(N_, false));

            // set up initial conditions for qk1_gpu and qk2_gpu = FFT(h)
            thrust::copy(h_gpu_.begin(), h_gpu_.end(), q1_[1]);
            cufftDoubleComplex *V1 = (cufftDoubleComplex*)thrust::raw_pointer_cast(&(q1_[1][0]));
            GPU_ERR(cufftExecZ2Z(FFT_plan_, V1, V1, cuFFTFORWARD));

            // step through to get propagators in Fourier space.
            for (int i=1; i<N_; i++) Step_->fwd(q1_[i], q1_[i+1], hA_gpu_, i, true);

            // get the partition function
            Q_ = q1_[N_][0];
            Q_ /= M_;

            // zero the summation array (complex)
            thrust::fill(SUM_gpu_.begin(), SUM_gpu_.end(), cmplx_zero);

            // 
            for (i=1; i<N_; i++) { 
                auto minusK_iter_ = thrust::make_permutation_iterator(q2_[i+1], (*minusK_gpu_).begin()); 
                auto z2 = thrust::make_zip_iterator(thrust::make_tuple(SUM_gpu_.begin(), q1_[i], minusK_iter_));
                thrust::for_each(z2, z2+M_, sum_prod_functor());
            }

            // Do reduction summations
            for (i=0; i<3; i++) {
                auto z3 = thrust::make_zip_iterator(thrust::make_tuple(c_dQdL_gpu_.begin()+i*M_, Step_->g_gpu(), SUM_gpu_.begin()));
                dQdL[i] = thrust::transform_reduce(z3, z3 + M_, dQdL_functor(), cmplx_zero, thrust::plus<thrust::complex<double>>());
                dQdL[i] /= L_[i]*L_[i]*L_[i]*M_*M_;
            }

            return Q_;
        }





        // Destructor
        ~diblockClass() {
            delete Step_;
        }





    private:
        // Calculate the wavevector moduli and store in K[]
        void calcK(double *c_dQdL, int *m, double *L) {
            int K0, K1, K2, k;
            //int mK0, mK1, mK2;

            for (int k0=-(m[0]-1)/2; k0<=m[0]/2; k0++) {
                K0 = (k0<0)?(k0+m[0]):k0;
                //mK0 = (k0>0)?(-k0+m[0]):-k0;

                for (int k1=-(m[1]-1)/2; k1<=m[1]/2; k1++) {
                    K1 = (k1<0)?(k1+m[1]):k1;
                    //mK1 = (k1>0)?(-k1+m[1]):-k1;

                    for (int k2=-(m[2]-1)/2; k2<=m[2]/2; k2++) {
                        K2 = (k2<0)?(k2+m[2]):k2;
                        //mK2 = (k2>0)?(-k2+m[2]):-k2;

                        k = K2+m[2]*(K1+m[1]*K0);
                        //minusK[k] = mK2+m[2]*(mK1+m[1]*mK0);

                        c_dQdL[k] = (4*M_PI*M_PI*k0*k0)*M_/(3*N_);
                        c_dQdL[k+M_] = (4*M_PI*M_PI*k1*k1)*M_/(3*N_);
                        c_dQdL[k+2*M_] = (4*M_PI*M_PI*k2*k2)*M_/(3*N_);
                    }
                }
            }
        }

};

// ######################################################
// Provides public method: calc_concs(double *w_gpu), 
// to calculate concentrations (used in Anderson mixing)
// ######################################################

#pragma once
#include <cuda.h>
#include "GPUkernels.h"
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







class diblockClass {

    // Diblock-specific variables
    int TpB_;
    step *Step_;                                                                            // Step object to get propagators for the next monomer

    thrust::device_vector<thrust::complex<double>> qr_gpu_;                                 // Pointer to GPU memory for propagators: q_{i}(r) and q^_{N+1-i}(r) are contigious in memory

    thrust::device_vector<thrust::complex<double>> h_gpu_;                                  // GPU memory for hA(r) and hB(r)
    thrust::device_vector<thrust::complex<double>>::iterator hA_gpu_, hB_gpu_;

    thrust::host_vector<thrust::device_vector<thrust::complex<double>>::iterator> q1_;      // Array of pointers to q_{j=i}(r), where j is the monomer index and i is array index
    thrust::host_vector<thrust::device_vector<thrust::complex<double>>::iterator> q2_;      // Array of pointers to q^_{j=N+1-i}(r), where j is the monomer index and i is array index

    thrust::complex<double> Q_;                                                             // Partition function after the last call to calc_concs()

    // Simulation constants derived from the input file (see lfts_params.h for details)
    int M_;
    int NA_;
    int N_;



    public:
        // Constructor
        diblockClass(int NA, int NB, int* m, double* L, int M, int TpB = 512) {
            TpB_ = TpB;

            M_ = M;
            NA_ = NA;
            N_ = NA+NB;

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

            // New step object containing methods to get next monomer's propagators
            Step_ = new step(NA, NB, m, L, M);
        }

        // Returns the partition function from the most recent call to calc_concs()
        thrust::complex<double> Q() { return Q_; }


        // Calculates phi-(r) and phi+(r): w+2*M -> phi-(0), w+3*M -> phi+(0).
        // Returns ln(Q)
        thrust::complex<double> calc_concs(thrust::device_vector<thrust::complex<double>> &w_gpu) {
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

        // Destructor
        ~diblockClass() {
            delete Step_;
        }
};
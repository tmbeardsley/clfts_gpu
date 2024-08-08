// #############################################################################
// Performs a langevin update of w-(r) and w+(r)
// #############################################################################
#pragma once

#include <cuda.h>
#include <curand.h>
#include <thrust/extrema.h>
#include <thrust/iterator/constant_iterator.h>
#include "diblock.h"


//wm=get<0>(t)
//phim=get<1>(t)
//phip=get<2>(t)
//lambda1_m=get<3>(t)
//lambda1_p=get<4>(t)
//wp=get<5>(t)
struct get_lambda_functor
{
    const double XbN_, zeta_;
    get_lambda_functor(double XbN, double zeta) : XbN_(XbN), zeta_(zeta) {}
    template <typename Tuple>
    __host__ __device__
        void operator()(Tuple t) {

        // w-
        thrust::get<3>(t) = thrust::get<1>(t) + 2.0 * thrust::get<0>(t) / XbN_;

        // w+
        thrust::get<4>(t) = thrust::get<2>(t) - 1.0 - 2.0 * thrust::get<5>(t) / (XbN_ + 2.0 * zeta_);
    }
};

//wm=get<0>(t)
//wp=get<1>(t)
//sym_noise=get<2>(t)
//isym_noise=get<3>(t)
//lambda1_m=get<4>(t)
//lambda1_p=get<5>(t)
struct langevin_P_functor
{
    const double dt_;
    const double lam_m_, lam_p_;
    thrust::complex<double> I_ = { 0.0,1.0 };
    langevin_P_functor(double dt, double lam_m, double lam_p) : dt_(dt), lam_m_(lam_m), lam_p_(lam_p) {}
    template <typename Tuple>
    __host__ __device__
        void operator()(Tuple t) {

        // predictor w-
        thrust::get<0>(t) += -thrust::get<4>(t)*lam_m_*dt_ + thrust::get<2>(t);

        // predictor w+
        thrust::get<1>(t) += thrust::get<5>(t)*lam_p_*dt_ + I_*thrust::get<3>(t);
    }
};

//wm=get<0>(t)
//wp=get<1>(t)
//phim=get<2>(t)
//phip=get<3>(t)
//sym_noise=get<4>(t)
//isym_noise=get<5>(t)
//lambda1_m=get<6>(t)
//lambda1_p=get<7>(t)
//w_cpy_m=get<8>(t)
//w_cpy_p=get<9>(t)
struct langevin_C_functor
{
    const double dt_;
    const double XbN_, zeta_;
    const double lam_m_, lam_p_;
    thrust::complex<double> I_ = { 0.0,1.0 };
    langevin_C_functor(double XbN, double dt, double lam_m, double lam_p, double zeta) : XbN_(XbN), dt_(dt), lam_m_(lam_m), lam_p_(lam_p), zeta_(zeta) {}
    template <typename Tuple>
    __host__ __device__
        void operator()(Tuple t) {
        thrust::complex<double> lambda2;

        // corrector w-
        lambda2 = thrust::get<2>(t) + 2.0 * thrust::get<0>(t) / XbN_;
        thrust::get<0>(t) = thrust::get<8>(t) - 0.5 * (thrust::get<6>(t) + lambda2)*lam_m_*dt_ + thrust::get<4>(t);

        // corrector w+
        lambda2 = thrust::get<3>(t) - 1.0 - 2.0 * thrust::get<1>(t) / (XbN_ + 2.0*zeta_);
        thrust::get<1>(t) = thrust::get<9>(t) + 0.5 * (thrust::get<7>(t) + lambda2)*lam_p_*dt_ + I_*thrust::get<5>(t);
    }
};

struct compare_cmplx_value
{
    __host__ __device__
        bool operator()(thrust::complex<double> a, thrust::complex<double> b)
    {
        return thrust::norm(a) < thrust::norm(b);
    }
};

//============================================================
// sign function
//------------------------------------------------------------
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}







class langevin_cmplx {
    thrust::device_vector<double> noise_vec_;
    thrust::device_vector<thrust::complex<double>> lambda1_;
    thrust::device_vector<thrust::complex<double>> w_cpy_;


    // Simulation constants derived from the input file (see lfts_params.h for details)
    int     M_;
    double  dt_;    // nominal time step

    double adt_;
    double K_ATS_;
    double dK_ATS_;

    // Mobility factors
    double lam_m_ = 1.0;
    double lam_p_ = 1.0;

    public:
        langevin_cmplx(curandGenerator_t& RNG, double dt, int M, double K_ATS = 2.8, double dK_ATS = 1E-4) {
            M_ = M;
            dt_ = dt;

            adt_ = dt;
            K_ATS_ = K_ATS;
            dK_ATS_ = dK_ATS;

            // Allocate memory for Gaussian random noise on the GPU
            noise_vec_.resize(2*M);

            // Allocate memory for the forcing term
            lambda1_.resize(2*M);

            // Allocate memory for copying the W-(r) and W+(r) fields
            w_cpy_.resize(2*M);
        }

        ~langevin_cmplx() {
        }

        double step_wm_wp(
            thrust::device_vector<thrust::complex<double>>& w_gpu,
            diblockClass* dbc,      // Diblock class for calculating concentrations
            curandGenerator_t& gen,
            double sigma,           // sigma passed for future box move (could be non-constant)
            double XbN,             // XbN passed for thermodynamic integration
            double zeta = 1E10,		// approximately no compressibility
            bool adapt_K_ATS = false,
            bool ATS = false,
            bool RTN_GMAX = false) {

            thrust::complex<double> lnQ;
            double G_max_abs = -1;

            // set adaptive time step = nominal time step initially
            adt_ = dt_;

            // make a copy of the original w- and w+ fields
            thrust::copy(w_gpu.begin(), w_gpu.begin() + 2*M_, w_cpy_.begin());

            // get the forcing terms for w- and w+
            calculate_force(lambda1_, w_gpu, XbN, zeta);

            // get the largest absolute value of the force
            if (RTN_GMAX || ATS) {
                thrust::complex<double> lam_max = *(thrust::max_element(lambda1_.begin(), lambda1_.end(), compare_cmplx_value()));
                G_max_abs = thrust::abs(lam_max);
            }

            // calculate size of the adaptive time step
            if (ATS) adt_ = (K_ATS_ / G_max_abs) * dt_;

            // get scaled noise for langevin step
            curandGenerateNormalDouble(gen, (double*)thrust::raw_pointer_cast(&(noise_vec_[0])), M_, 0.0, sigma*sqrt(lam_m_*adt_/dt_));
            curandGenerateNormalDouble(gen, (double*)thrust::raw_pointer_cast(&(noise_vec_[M_])), M_, 0.0, sigma*sqrt(lam_p_*adt_/dt_));

            // perform predictor langevin step (pass the pre-computed forcing terms)
            auto zP = thrust::make_zip_iterator(
                        thrust::make_tuple(
                            w_gpu.begin(),
                            w_gpu.begin() + M_,
                            noise_vec_.begin(),
                            noise_vec_.begin() + M_,
                            lambda1_.begin(),
                            lambda1_.begin() + M_));
            thrust::for_each(zP, zP+M_, langevin_P_functor(adt_, lam_m_, lam_p_));

            // calculate phi- and phi+ from predicted fields
            lnQ = dbc->calc_concs(w_gpu);

            // perform corrector langevin step
            auto zC = thrust::make_zip_iterator(
                        thrust::make_tuple(
                            w_gpu.begin(),
                            w_gpu.begin() + M_,
                            w_gpu.begin() + 2*M_,
                            w_gpu.begin() + 3*M_,
                            noise_vec_.begin(),
                            noise_vec_.begin() + M_,
                            lambda1_.begin(),
                            lambda1_.begin() + M_,
                            w_cpy_.begin(),
                            w_cpy_.begin() + M_));
            thrust::for_each(zC, zC + M_, langevin_C_functor(XbN, adt_, lam_m_, lam_p_, zeta));

            // calculate phi- and phi+ from predicted fields
            lnQ = dbc->calc_concs(w_gpu);

            // Update K_ATS (controls size of updates to adaptive time step)
            if (adapt_K_ATS) K_ATS_ += sgn(dt_ - adt_) * dK_ATS_;

            return G_max_abs;
        }

        // Getters
        double adt() { return adt_; }
        double K_ATS() { return K_ATS_; }



    private:
        // Note: XbN passed to function to allow for future thermodynamic integration modifications
        void calculate_force(thrust::device_vector<thrust::complex<double>>& lambda1, thrust::device_vector<thrust::complex<double>>& w_gpu, double XbN, double zeta) {

            // zip iterator for the get_lambda_functor()
            auto z = thrust::make_zip_iterator(
                        thrust::make_tuple(
                            w_gpu.begin(),
                            w_gpu.begin() + 2*M_,
                            w_gpu.begin() + 3*M_,
                            lambda1.begin(),
                            lambda1.begin() + M_,
                            w_gpu.begin() + M_));

            // get the forcing terms for w- and w+
            thrust::for_each(z, z+M_, get_lambda_functor(XbN, zeta));
        }



};

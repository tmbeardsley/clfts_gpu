#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>
#include <thrust/transform.h>

typedef thrust::complex<double> cmplx_dbl;

using namespace std;


// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square_unary_op
{
    __host__ __device__
    T operator()(T a)
    {
        return a*a;
    }
};


struct compare_cmplx_value
{
	__host__ __device__
    bool operator()(cmplx_dbl a, cmplx_dbl b)
	{
		return thrust::norm(a) < thrust::norm(b);
	}
};



//wm=get<0>(t)
//phim=get<1>(t)
//phip=get<2>(t)
//lambda1_m=get<3>(t)
//lambda1_p=get<4>(t)
//wp=get<5>(t)
struct get_lambda_functor
{
    const double chi_b, zeta;
    get_lambda_functor(double _chi_b, double _zeta) : chi_b(_chi_b), zeta(_zeta) {}
    template <typename Tuple>
    __host__ __device__
        void operator()(Tuple t) {

        // w-
        thrust::get<3>(t) = thrust::get<1>(t) + 2.0*thrust::get<0>(t)/chi_b;

        // w+
        thrust::get<4>(t) = thrust::get<2>(t) - 1.0 - 2.0*thrust::get<5>(t)/(chi_b+2.0*zeta);
    }
};



//wm=get<0>(t) Y
//wp=get<1>(t) Y
//sym_noise=get<2>(t) Y
//isym_noise=get<3>(t) Y
//lambda1_m=get<4>(t) Y
//lambda1_p=get<5>(t) Y
struct langevin_P_functor
{
    const double dt;
    const double lam_m, lam_p;
    thrust::complex<double> I = { 0.0,1.0 };
    langevin_P_functor(double _dt, double _lam_m, double _lam_p) : dt(_dt), lam_m(_lam_m), lam_p(_lam_p) {}
    template <typename Tuple>
    __host__ __device__
        void operator()(Tuple t) {  

        // predictor w-
        thrust::get<0>(t) += -thrust::get<4>(t)*lam_m*dt + thrust::get<2>(t);

        // predictor w+
        thrust::get<1>(t) += thrust::get<5>(t)*lam_p*dt + I*thrust::get<3>(t);
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
    const double dt;
    const double chi_b, zeta;
    const double lam_m, lam_p;
    thrust::complex<double> I = {0.0,1.0};
    langevin_C_functor(double _chi_b, double _dt, double _lam_m, double _lam_p, double _zeta) : chi_b(_chi_b), dt(_dt), lam_m(_lam_m), lam_p(_lam_p), zeta(_zeta) {}
    template <typename Tuple>
    __host__ __device__
        void operator()(Tuple t) {
        thrust::complex<double> lambda2;

        // corrector w-
        lambda2 = thrust::get<2>(t) + 2.0*thrust::get<0>(t)/chi_b;
        thrust::get<0>(t) = thrust::get<8>(t) - 0.5*(thrust::get<6>(t)+lambda2)*lam_m*dt + thrust::get<4>(t);

        // corrector w+
        lambda2 = thrust::get<3>(t) - 1.0 - 2.0*thrust::get<1>(t)/(chi_b+2.0*zeta);
        thrust::get<1>(t) = thrust::get<9>(t) + 0.5*(thrust::get<7>(t)+lambda2)*lam_p*dt + I*thrust::get<5>(t);
    }
};

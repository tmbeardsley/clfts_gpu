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

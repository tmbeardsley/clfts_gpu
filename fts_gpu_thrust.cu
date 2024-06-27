// fts.cu
//------------------------------------------------------------
// GPU version of the FTS code for a diblock copolymer melt
// Note that lengths are expressed in units of R0=a*N^0.5
//------------------------------------------------------------

#include <math.h>        // math subroutines
#include <stdlib.h>      // standard library
#include <time.h>        // required to seed random number generator
#include <complex>       // for complex-valued variables
#include <cuda.h>       // required for GPUs
#include <cuda_runtime_api.h>   // required for GPUs
#include <iostream>
#include <fstream>
#include "GPUerror.h"   // GPU error handling kernels
#include "GPUkernels.h" // GPU kernels
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/transform_reduce.h>
#include <curand.h>
#include <sys/stat.h>
#include "step.h"
#include "diblock.h"
#include "clfts_params.h"
#include "file_IO.h"
#include "langevin_cmplx.h"
#include "strFuncCmplx.h"
#include "order_parameter.h"
#include <iomanip>
using namespace std;

typedef thrust::host_vector<thrust::complex<double>> hvec_cmplx;
typedef thrust::device_vector<thrust::complex<double>> dvec_cmplx;
typedef thrust::device_vector<double> dvec_dbl;


//==============================================================
// Write an array to file
//--------------------------------------------------------------
void write_array_to_file(string file_mask, int step_num, complex<double>* arr, int _n) {
    ofstream out_stream;

    // write the field to file
    out_stream.open(file_mask + std::to_string(step_num));
    out_stream.precision(10);
    for (int r = 0; r < _n; r++) out_stream << arr[r].real() << "\t" << arr[r].imag() << endl;
    out_stream.close();
}









//------------------------------------------------------------
int main()
{
    thrust::complex<double> I = { 0.0, 1.0 };
    double chi_b, chi_e, zeta, L[3], dt, C, sigma;
    thrust::complex<double> lnQ, wm, wm_sq, wp, Hf;
    thrust::host_vector<thrust::complex<double>> dQdL(3);
    int    r, N;
    int    it, equil_its, sim_its, sample_freq;
    FILE* out;
    int WRITE_FREQ = 50000;
    int OUT_FREQ = 100;
    double dK_ATS = 1E-4;
    int seed;
    int    m[3], M, NA, NB;
    diblockClass* dbc;

    cout << "Creating clfts_params()..." << endl;
    clfts_params* P = new clfts_params("input");
    cout << "clfts_params() created!" << endl;
    P->outputParameters();

    // Get parameters
    N = P->N();
    NA = P->NA();
    chi_e = P->XeN();
    chi_b = P->XbN();
    zeta = P->zeta();
    C = P->C();
    dt = P->dt();
    m[0] = P->mx();
    m[1] = P->my();
    m[2] = P->mz();
    L[0] = P->Lx();
    L[1] = P->Ly();
    L[2] = P->Lz();
    equil_its = P->equil_its();
    sim_its = P->sim_its();
    sample_freq = P->sample_freq();
    M = P->M();
    sigma = P->sigma();
    NB = P->NB();

    cout << "Creating diblock()..." << endl;
    dbc = new diblockClass(NA, NB, m, L, M);
    cout << "diblock() created!" << endl;




    // declare w on the host and gpu
    hvec_cmplx w(4*M);
    dvec_cmplx w_gpu(4*M);

    // Create and seed pseudo-random number generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    seed = 123456789;
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    // read fields from input file 
    file_IO::readCmplxVector(w, "input", 2 * M, 3);

    // write input to checkpoint file for easier programming when loading from numerical crash
    write_array_to_file("Wm_eq_", 0, (complex<double>*)thrust::raw_pointer_cast(&((w)[0])), M);
    write_array_to_file("Wp_eq_", 0, (complex<double>*)thrust::raw_pointer_cast(&((w)[M])), M);

    // copy host w to w_gpu after reading input file
    w_gpu = w;

    // call the calc_concs subroutine to get the 
    lnQ = dbc->calc_concs(w_gpu);
    cout << endl << "lnQ_orig = " << lnQ << endl;

    // Tunable parameter for the adaptive time step
    double K_ATS = 2.8;
    cout << endl << "K_ATS = " << K_ATS << endl << endl;










    cout << "Creating langevin_cmplx()..." << endl;
    langevin_cmplx Langevin(gen, dt, M, K_ATS, dK_ATS);
    cout << "langevin_cmplx() created!" << endl;


    cout << "Creating strFuncCmplx()..." << endl;
    strFuncCmplx *Sk = new strFuncCmplx(m, L, M, P->n(), P->XbN());
    cout << "strFuncCmplx() created!" << endl;

    cout << "Creating order_parameter()..." << endl;
    order_parameter *Psi = new order_parameter(m, L, M);
    cout << "order_parameter() created!" << endl;

    // titles for outputs
    cout << "it" << "\t"
        << "lnQ.r" << "\t"
        << "lnQ.i" << "\t"
        << "wm.r" << "\t"
        << "wm.i" << "\t"
        << "wm_sq.r" << "\t"
        << "wm_sq.i" << "\t"
        << "wp.r" << "\t"
        << "wp.i" << "\t"
        << "Hf.r" << "\t"
        << "Hf.i" << "\t"
        << endl;






    // equilibrate the system
    time_t t_start, t_end;
    t_start = time(NULL);
    for (it = 1; it <= equil_its; it++) {

        Langevin.step_wm_wp(w_gpu, dbc, gen, sigma, chi_b, zeta, true, true, true);

        if (it % sample_freq == 0) {
            // re-compute partition function
            lnQ = dbc->calc_concs(w_gpu);

            // compute averages
            wm = thrust::reduce(w_gpu.begin(), w_gpu.begin() + M);
            wp = thrust::reduce(w_gpu.begin() + M, w_gpu.begin() + 2 * M);
            wm_sq = 0.0 + I * 0.0;
            wm_sq = thrust::transform_reduce(w_gpu.begin(), w_gpu.begin() + M, square_unary_op<thrust::complex<double>>(), wm_sq, thrust::plus<thrust::complex<double>>());

            Hf = -lnQ + wm_sq / (chi_b * M) - wp / M;   // Hf = -log(Q)+0.25*chi_b+wm2/chi_b-wp2/(chi_b+2.0/kappa)-wp1;


            if (it % OUT_FREQ == 0) {
                std::cout << it << "\t"
                    << lnQ.real() << "\t" << lnQ.imag() << "\t"
                    << wm.real() / M << "\t" << wm.imag() / M << "\t"
                    << wm_sq.real() / M << "\t" << wm_sq.imag() / M << "\t"
                    << wp.real() / M << "\t" << wp.imag() / M << "\t"
                    << Hf.real() << "\t" << Hf.imag() << "\t"
                    << dbc->Q() << "\t" << dbc->get_Q_derivatives(w_gpu, dQdL) << "\t"
                    << Psi->get_Psi4(w_gpu)
                    << endl;
                std::cout   << (L[0]*dQdL[0]/dbc->Q()).real() << "\t"
                            << (L[1]*dQdL[1]/dbc->Q()).real() << "\t"
                            << (L[2]*dQdL[2]/dbc->Q()).real() << "\t"
                            << endl;
            }
        }

        if (it % WRITE_FREQ == 0) {
            // re-compute partition function
            lnQ = dbc->calc_concs(w_gpu);

            // write fields to file (copy from gpu to cpu)
            w = w_gpu;
            write_array_to_file("Wm_eq_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[0])), M);
            write_array_to_file("Wp_eq_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[M])), M);
            write_array_to_file("PHIm_eq_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[2 * M])), M);
            write_array_to_file("PHIp_eq_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[3 * M])), M);
        }

    }
    t_end = time(NULL);
    printf("equlibration time = %ld secs\n", t_end - t_start);
    cout << endl << "K_ATS = " << K_ATS << endl << endl;









    // save checkpoint
    w = w_gpu;
    write_array_to_file("Wm_eq_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[0])), M);
    write_array_to_file("Wp_eq_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[M])), M);








    t_start = time(NULL);
    for (it = 1; it <= sim_its; it++) {

        Langevin.step_wm_wp(w_gpu, dbc, gen, sigma, chi_b, zeta, false, true, true);

        if (it % sample_freq == 0) {
            // re-compute partition function
            lnQ = dbc->calc_concs(w_gpu);

            // compute averages
            wm = thrust::reduce(w_gpu.begin(), w_gpu.begin() + M);
            wp = thrust::reduce(w_gpu.begin() + M, w_gpu.begin() + 2 * M);
            wm_sq = 0.0 + I * 0.0;
            wm_sq = thrust::transform_reduce(w_gpu.begin(), w_gpu.begin() + M, square_unary_op<thrust::complex<double>>(), wm_sq, thrust::plus<thrust::complex<double>>());

            Hf = -lnQ + wm_sq / (chi_b * M) - wp / M;

            Sk->sample(w_gpu);

            if (it % OUT_FREQ == 0) {
                cout << it << "\t"
                    << lnQ.real() << "\t" << lnQ.imag() << "\t"
                    << wm.real() / M << "\t" << wm.imag() / M << "\t"
                    << wm_sq.real() / M << "\t" << wm_sq.imag() / M << "\t"
                    << wp.real() / M << "\t" << wp.imag() / M << "\t"
                    << Hf.real() << "\t" << Hf.imag() << "\t"
                    << dbc->Q() << "\t" << dbc->get_Q_derivatives(w_gpu, dQdL) << "\t"
                    << Psi->get_Psi4(w_gpu)
                    << endl;
                std::cout   << (L[0]*dQdL[0]/dbc->Q()).real() << "\t"
                            << (L[1]*dQdL[1]/dbc->Q()).real() << "\t"
                            << (L[2]*dQdL[2]/dbc->Q()).real() << "\t"
                            << endl;

                Sk->save("Sk_" + to_string(it));
            }
        }

        if (it % WRITE_FREQ == 0) {
            // re-compute partition function
            lnQ = dbc->calc_concs(w_gpu);

            // write fields to file (copy from gpu to cpu)
            w = w_gpu;
            write_array_to_file("Wm_st_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[0])), M);
            write_array_to_file("Wp_st_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[M])), M);
            write_array_to_file("PHIm_st_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[2 * M])), M);
            write_array_to_file("PHIp_st_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[3 * M])), M);
        }
    }
    t_end = time(NULL);
    printf("statistics time = %ld secs\n", t_end - t_start);





    // copy data from GPU to CPU for output
    w = w_gpu;
    out = fopen("output", "w");
    fprintf(out, "%d %d %lf %lf %lf %lf\n", N, NA, chi_e, zeta, C, dt);
    fprintf(out, "%d %d %d %lf %lf %lf\n", m[0], m[1], m[2], L[0], L[1], L[2]);
    fprintf(out, "%d %d %d\n", equil_its, sim_its, sample_freq);
    for (r = 0; r < 2 * M; r++) fprintf(out, "%lf\t%lf\n", w[r].real(), w[r].imag());
    fclose(out);




    delete dbc;
    delete P;
    delete Sk;
    delete Psi;
    return 0;

}

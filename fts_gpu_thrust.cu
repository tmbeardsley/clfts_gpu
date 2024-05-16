// fts.cu
//------------------------------------------------------------
// GPU version of the FTS code for a diblock copolymer melt
// Note that lengths are expressed in units of R0=a*N^0.5
//------------------------------------------------------------

#include<math.h>        // math subroutines
#include<stdlib.h>      // standard library
#include<time.h>        // required to seed random number generator
#include<complex>       // for complex-valued variables
#include <cuda.h>       // required for GPUs
#include <cuda_runtime_api.h>   // required for GPUs
//#include <cufft.h>      // required for fast Fourier transforms
#include<iostream>
#include<fstream>
#include "GPUerror.h"   // GPU error handling kernels
#include "GPUkernels.h" // GPU kernels
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/zip_function.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <curand.h>
#include <sys/stat.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/extrema.h>
#include "step.h"
#include "diblock.h"
#include "clfts_params.h"
#include <iomanip>
#define cuFFTFORWARD -1
#define cuFFTINVERSE 1
using namespace std;

typedef thrust::host_vector<thrust::complex<double>> hvec_cmplx;
typedef thrust::device_vector<thrust::complex<double>> dvec_cmplx;
typedef thrust::device_vector<thrust::complex<double>>::iterator dvec_cmplx_itr;
typedef thrust::host_vector<thrust::complex<double>>::iterator hvec_cmplx_itr;
typedef thrust::device_vector<double> dvec_dbl;
typedef thrust::complex<double> cmplx_dbl;

diblockClass* dbc;


//------------------------------------------------------------
// Global CPU variables:
//
// m[i] = number of mesh points in i'th dimension
// M = m[0]*m[1]*m[2] = total number of mesh points
// NA and NB = number of monomers in A and B blocks

int    m[3], M, NA, NB;
double two_pi=8*atan(1.0), pi=4*atan(1.0);
ofstream dbg_stream;

//------------------------------------------------------------
// Global GPU variables:
//
// w_gpu[4*M] = GPU copy of w[4*M] in main
// q1_N[r] = q_N(r)



//==============================================================
// Write an array to file
//--------------------------------------------------------------
void write_array_to_file(string file_mask, int step_num, complex<double>* arr, int _n) {
    ofstream out_stream;

    // write the field to file
    out_stream.open(file_mask + std::to_string(step_num));
    out_stream.precision(10);
    for (int r=0; r<_n; r++) out_stream << arr[r].real() << "\t" << arr[r].imag() << endl;
    out_stream.close();
}
//==============================================================
// calculates z_infty for discrete polymer chain 
//--------------------------------------------------------------
double z_inf_discrete(double* L, double Nbar) {
    int tmax, N = NA + NB;
    double sum, X, l, L1, L2, L3, R0;

    tmax = 100;
    sum = 0.5;
    R0 = sqrt((double)N);
    L1 = R0 * L[0] / m[0];
    L2 = R0 * L[1] / m[1];
    L3 = R0 * L[2] / m[2];
    l = pow(L1 * L2 * L3, 1.0 / 3.0);
    for (int t = 1; t <= tmax; t++) {
        X = (pi / l) * sqrt(t / 6.0);
        sum += pow(sqrt(pi) / (2 * X), 3.0) * erf(X * l / L1) * erf(X * l / L2) * erf(X * l / L3);
    }
    sum *= 2 * R0 / (pow(l, 3.0) * sqrt(Nbar));
    X = (pi / l) * sqrt((0.5 + tmax) / 6.0);

    return 1.0 - sum - 3 * R0 / (l * sqrt(pi * Nbar) * X);
}




//============================================================
// perform a PC Langevin step
//------------------------------------------------------------
double langevin_PC_adt( dvec_cmplx& w_gpu,
                    curandGenerator_t& gen,
                    dvec_dbl& noise_vec,
                    thrust::complex<double>& lnQ,
                    dvec_cmplx &lambda1,
                    dvec_cmplx &w_cpy,
                    double sigma,
                    double chi_b,
                    double dt,
                    double &adt,
		            double zeta = 1E10,		// approximately no compressibility
                    bool RTN_GMAX = false,
                    bool ATS=false,
                    double K_ATS=1.0)
{
    // set up temporary vectors and pointers
    dvec_cmplx_itr lambda1_m, lambda1_p, w_cpy_m, w_cpy_p;

    lambda1_m = lambda1.begin(); lambda1_p = lambda1_m + M;
    w_cpy_m = w_cpy.begin(); w_cpy_p = w_cpy_m + M;

    double lam_m=1.0, lam_p=1.0, G_max_abs=0.0;
    double G_max_abs_m, G_max_abs_p;

    // set adaptive time step = nominal time step to begin with
    adt = dt;

    // make a copy of the original w- and w+ fields
    thrust::copy(w_gpu.begin(), w_gpu.begin()+2*M, w_cpy.begin());

    // get the forcing terms for w- and w+
    thrust::for_each(   thrust::make_zip_iterator(  thrust::make_tuple( w_gpu.begin(),
                                                                        w_gpu.begin() + 2*M,
                                                                        w_gpu.begin() + 3*M,
                                                                        lambda1_m,
                                                                        lambda1_p,
									                                    w_gpu.begin() + M)),
                        thrust::make_zip_iterator(  thrust::make_tuple( w_gpu.begin() + M,
                                                                        w_gpu.begin() + 3*M,
                                                                        w_gpu.begin() + 4*M,
                                                                        lambda1_m + M,
                                                                        lambda1_p + M,
									                                    w_gpu.begin() + 2*M)),
                        get_lambda_functor(chi_b, zeta));



    if (RTN_GMAX || ATS) {
        // get an iterator to the max of lambda1_m and lambda1_p
        dvec_cmplx_itr lam_max_itr = thrust::max_element(lambda1.begin(), lambda1.begin()+M, compare_cmplx_value());

        // get the max value of the forcing term (G_max)
        //cout << *lam_max_itr << " @ " << lam_max_itr - lambda1.begin() << endl;
        cmplx_dbl lam_max = *lam_max_itr;

        // get size of adaptive time step
        G_max_abs_m = thrust::abs(lam_max);

        // get an iterator to the max of lambda1_m and lambda1_p
        lam_max_itr = thrust::max_element(lambda1.begin()+M, lambda1.end(), compare_cmplx_value());
        lam_max = *lam_max_itr;
        G_max_abs_p = thrust::abs(lam_max);

	    G_max_abs = max(G_max_abs_m, G_max_abs_p);

        //cout << G_max_abs_m << "\t" << G_max_abs_p << "\t" << G_max_abs << endl;
    }




    if (ATS) {
        // calculate size of the adaptive time step
        adt = (K_ATS/G_max_abs)*dt;
        //cout << adt << endl;
    }


    // get sigma- and sigma+
    double sigma_m = sigma*sqrt(lam_m*adt/dt);
    double sigma_p = sigma*sqrt(lam_p*adt/dt);

    // get scaled noise for langevin step
    curandGenerateNormalDouble(gen, (double*)thrust::raw_pointer_cast(&(noise_vec[0])), M, 0.0, sigma_m);
    curandGenerateNormalDouble(gen, (double*)thrust::raw_pointer_cast(&(noise_vec[M])), M, 0.0, sigma_p);


    // perform predictor langevin step (pass the pre-computed forcing terms)
    thrust::for_each(   thrust::make_zip_iterator(  thrust::make_tuple( w_gpu.begin(),
                                                                        w_gpu.begin() + M,
                                                                        noise_vec.begin(),
                                                                        noise_vec.begin()+M,
                                                                        lambda1_m,
                                                                        lambda1_p)),
                        thrust::make_zip_iterator(  thrust::make_tuple( w_gpu.begin() + M,
                                                                        w_gpu.begin() + 2*M,
                                                                        noise_vec.begin() + M,
                                                                        noise_vec.begin() + 2*M,
                                                                        lambda1_m + M,
                                                                        lambda1_p + M)),
                        langevin_P_functor(adt, lam_m, lam_p));

    // calculate phi- and phi+ from predicted fields
    lnQ = dbc->calc_concs(w_gpu);

    // perform corrector langevin step
    thrust::for_each(   thrust::make_zip_iterator(  thrust::make_tuple( w_gpu.begin(),
                                                                        w_gpu.begin() + M,
                                                                        w_gpu.begin() + 2*M,
                                                                        w_gpu.begin() + 3*M,
                                                                        noise_vec.begin(),
                                                                        noise_vec.begin()+M,
                                                                        lambda1_m,
                                                                        lambda1_p,
                                                                        w_cpy_m,
                                                                        w_cpy_p)),
                        thrust::make_zip_iterator(  thrust::make_tuple( w_gpu.begin() + M,
                                                                        w_gpu.begin() + 2*M,
                                                                        w_gpu.begin() + 3*M,
                                                                        w_gpu.begin() + 4*M,
                                                                        noise_vec.begin() + M,
                                                                        noise_vec.begin() + 2*M,
                                                                        lambda1_m + M,
                                                                        lambda1_p + M,
                                                                        w_cpy_m + M,
                                                                        w_cpy_p + M)),
                        langevin_C_functor(chi_b, adt, lam_m, lam_p, zeta));


    // calculate phi- and phi+ from corrected fields
    lnQ = dbc->calc_concs(w_gpu);

    // get the average of w+
    thrust::complex<double> wp_avg = thrust::reduce(w_gpu.begin() + M, w_gpu.begin() + 2*M)/M;

    // subtract <w+> from the w+ field
    thrust::transform(w_gpu.begin() + M, w_gpu.begin() + 2*M, thrust::make_constant_iterator(wp_avg), w_gpu.begin() + M, thrust::minus<thrust::complex<double>>());


    return G_max_abs;
}

//============================================================
// sign function
//------------------------------------------------------------
template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}







//============================================================
// Key variables in main:
//
// L[i] = size of the simulation box in the i'th dimension
// V = L[0]*L[1]*L[2] = volume of the simulation box
// w[4*M] = array containing N*w-, N*w+, phi-, phi+
// r = (x*m[1]+y)*m[2]+z = array position for (x,y,z)
// N = total number of monomers 
// chi_b = bare chi times N
// C = sqrt(Nbar) = dimensionless concentration
// sigma = standard deviation of the random noise times N
// dt = size of the Langevin step times N
// Hf = Hamiltonian in units of nkT
// lnQ = log of the single-chain partition function
// S[k] = structure function
// wt[k] = weighting of the wavevectors (equals 1 or 2)
// equil_its = number of equilibration steps
// sim_its = number of simulation steps
// sample_freq = frequency that observables are sampled
// wk[k] = fourier transform of w

//------------------------------------------------------------
int main ()
{
  thrust::complex<double> I={0.0,1.0};
  double chi_b, chi_e, zeta, L[3], V, dt, C, sigma, adt;
  thrust::complex<double> lnQ, wm, wm_sq, wp, Hf;
  double wR, wI;
  int    r, N;
  int    it, equil_its, sim_its, sample_freq;
  FILE *in, *out;
  int WRITE_FREQ = 50000;
  int OUT_FREQ = 100;
  double dK_ATS = 1E-4;
  int seed;

  cout << "Creating clfts_params()..." << endl;
  clfts_params *P = new clfts_params("input");
  cout << "clfts_params() created!" << endl;
  P->outputParameters();


  in = fopen("input","r");
  fscanf(in,"%d %d %lf %lf %lf %lf",&N,&NA,&chi_e,&zeta,&C,&dt);
  fscanf(in,"%d %d %d %lf %lf %lf",&m[0],&m[1],&m[2],&L[0],&L[1],&L[2]);
  fscanf(in,"%d %d %d",&equil_its,&sim_its,&sample_freq);

  sim_its = (sim_its/sample_freq)*sample_freq;  
  M = m[0]*m[1]*m[2];
  V = L[0]*L[1]*L[2];
  sigma = sqrt(2.0*M*dt/(C*V));
  NB = N-NA;


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
  V = P->V();
  sigma = P->sigma();
  NB = P->NB();







  cout << "Creating diblock()..." << endl;
  dbc = new diblockClass(NA, NB, m, L, M);
  cout << "diblock() created!" << endl;

  
  // declare w on the host and gpu
  hvec_cmplx w(4*M);
  dvec_cmplx w_gpu(4*M);

  // declare noise vectors: M real, M imag
  dvec_dbl noise_vec(2*M);

  // Create and seed pseudo-random number generator
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  seed = 123456789;
  curandSetPseudoRandomGeneratorSeed(gen, seed);


  // read fields from input file 
  for (r=0; r<2*M; r++) {
    fscanf(in,"%lf %lf", &wR, &wI);
    w[r].real(wR);
    w[r].imag(wI);
  }
  fclose(in);

  // write input to checkpoint file for easier programming when loading from numerical crash
  write_array_to_file("Wm_eq_", 0, (complex<double>*)thrust::raw_pointer_cast(&((w)[0])), M);
  write_array_to_file("Wp_eq_", 0, (complex<double>*)thrust::raw_pointer_cast(&((w)[M])), M);

  // copy host w to w_gpu after reading input file
  w_gpu = w;

  // set up the cufft plan for the fields
  dvec_dbl S_gpu(M);
  thrust::host_vector<double> S_out(M);

  // call the diblock subroutine
  lnQ = dbc->calc_concs(w_gpu);
  cout << endl << "lnQ_orig = " << lnQ << endl;

  // Get the tunable parameter for the adaptive time step
  bool ATS = true;
  double K_ATS = 2.8;
  cout << endl << "K_ATS = " << K_ATS << endl << endl;

  // device vectors required for langevin step
  dvec_cmplx lambda1(2*M), w_cpy(2*M);

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
  dbg_stream.open("dbg_stream.txt");
  time_t t_start, t_end;
  t_start = time(NULL);

    for (it = 1; it <= equil_its; it++) {

        langevin_PC_adt(w_gpu, gen, noise_vec, lnQ, lambda1, w_cpy, sigma, chi_b, dt, adt, zeta, true, true, K_ATS);
        K_ATS = K_ATS + sgn(dt - adt) * dK_ATS;
        if (it % 10 == 0) dbg_stream << it << "\t" << K_ATS << "\t" << adt << endl;

        if (it % sample_freq == 0) {
            // re-compute partition function
            lnQ = dbc->calc_concs(w_gpu);

            // compute averages
            wm = thrust::reduce(w_gpu.begin(), w_gpu.begin() + M);
            wp = thrust::reduce(w_gpu.begin() + M, w_gpu.begin() + 2 * M);
            wm_sq = 0.0 + I * 0.0;
            wm_sq = thrust::transform_reduce(w_gpu.begin(), w_gpu.begin() + M, square_unary_op<thrust::complex<double>>(), wm_sq, thrust::plus<thrust::complex<double>>());

            Hf = -lnQ + wm_sq / (chi_b * M) - wp / M;

            if (it % OUT_FREQ == 0) {
                cout << it << "\t"
                    << lnQ.real() << "\t" << lnQ.imag() << "\t"
                    << wm.real() / M << "\t" << wm.imag() / M << "\t"
                    << wm_sq.real() / M << "\t" << wm_sq.imag() / M << "\t"
                    << wp.real() / M << "\t" << wp.imag() / M << "\t"
                    << Hf.real() << "\t" << Hf.imag() << "\t"
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
  printf("equlibration time = %ld secs\n", t_end-t_start);
  cout << endl << "K_ATS = " << K_ATS << endl << endl;












  
  // reset number of reloads for statistics
  //n_reloads = 0;
  it = 0;

  // save initial checkpoint for loading if statistics fails
  w = w_gpu;
  write_array_to_file("Wm_st_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[0])), M);
  write_array_to_file("Wp_st_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[M])), M);

  // close current streams
  dbg_stream.close();



  dbg_stream.open("dbg_stream.txt");
  t_start = time(NULL);

    for (it=1; it<=sim_its; it++) {

        langevin_PC_adt(w_gpu, gen, noise_vec, lnQ, lambda1, w_cpy, sigma, chi_b, dt, adt, zeta, true, true, K_ATS);

        if (it%sample_freq == 0) {
            // re-compute partition function
            lnQ = dbc->calc_concs(w_gpu);

            // compute averages
            wm = thrust::reduce(w_gpu.begin(), w_gpu.begin()+M);
            wp = thrust::reduce(w_gpu.begin()+M, w_gpu.begin()+2*M);
            wm_sq = 0.0 + I*0.0;
            wm_sq = thrust::transform_reduce(w_gpu.begin(), w_gpu.begin()+M, square_unary_op<thrust::complex<double>>(), wm_sq, thrust::plus<thrust::complex<double>>());

            Hf = -lnQ + wm_sq/(chi_b*M) - wp/M;

            if (it%OUT_FREQ == 0) {
                cout << it << "\t"
                    << lnQ.real() << "\t" << lnQ.imag() << "\t"
                    << wm.real()/M << "\t" << wm.imag()/M << "\t"
                    << wm_sq.real()/M << "\t" << wm_sq.imag()/M << "\t"
                    << wp.real()/M << "\t" << wp.imag()/M << "\t"
                    << Hf.real() << "\t" << Hf.imag() << "\t"
                    << endl;
            }
        }

        if (it%WRITE_FREQ == 0) {
            // re-compute partition function
            lnQ = dbc->calc_concs(w_gpu);

            // write fields to file (copy from gpu to cpu)
            w = w_gpu;
            write_array_to_file("Wm_st_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[0])), M);
            write_array_to_file("Wp_st_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[M])), M);
            write_array_to_file("PHIm_st_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[2*M])), M);
            write_array_to_file("PHIp_st_", it, (complex<double>*)thrust::raw_pointer_cast(&((w)[3*M])), M);
        }
    }

  t_end = time(NULL);
  printf("statistics time = %ld secs\n", t_end - t_start);

  // close file streams
  dbg_stream.close();

  // copy data from GPU to CPU for output
  w = w_gpu;
  out = fopen("output", "w");
  fprintf(out, "%d %d %lf %lf %lf %lf\n", N, NA, chi_e, zeta, C, dt);
  fprintf(out, "%d %d %d %lf %lf %lf\n", m[0], m[1], m[2], L[0], L[1], L[2]);
  fprintf(out, "%d %d %d\n", equil_its, sim_its, sample_freq);
  for (r = 0; r < 2 * M; r++) fprintf(out, "%lf\t%lf\n", w[r].real(), w[r].imag());
  fclose(out);


  //delete Step;
  delete dbc;
  delete P;
  return 0;

}

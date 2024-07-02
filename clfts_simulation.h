#pragma once
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <thrust/transform_reduce.h>
#include <curand.h>
#include "step.h"
#include "diblock.h"
#include "clfts_params.h"
#include "file_IO.h"
#include "langevin_cmplx.h"
#include "strFuncCmplx.h"
#include "order_parameter.h"
#include "minusK_lookup.h"


// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square_op
{
    __host__ __device__
    T operator()(T a)
    {
        return a*a;
    }
};


class clfts_simulation {

    thrust::device_vector<thrust::complex<double>> w_gpu_;      // GPU array containing: N*w-(r), N*w+(r), phi-(r), phi+(r)
    thrust::host_vector<thrust::complex<double>> w_;            // Host array containing: N*w-(r), N*w+(r), phi-(r), phi+(r)
    clfts_params* P_;                                           // Object to hold the simulation parameters - automatically updates derived parameters
    diblockClass* dbc_;                                         // Diblock object for calculating phi-(r) and phi+(r)
    langevin_cmplx *Langevin_;                                  // Langevin object to update w-(r) and w+(r) at each step
    strFuncCmplx *Sk_;                                          // Object used for calculating the structure function
    order_parameter *Psi_;                                      // Object used to calculate the order parameter
    curandGenerator_t RNG_;                                     // Random number generator for the GPU
    minusK_lookup *minusK_lookup_;                              // Object holding the lookup table minusK(k)
    int M_;                                                     // Total number of field mesh points (constant - contained in lfts_params object but copied for tidier code)
    const int OUT_FREQ = 100;                                   // How often to output instantaneous values to screen during simulation


    public:
        clfts_simulation(std::string inputFile) {

            // Check that input files exist before proceeding
            std::string s = "";
            if (!file_IO::isValidFile(inputFile)) s += "ERROR => Cannot open the L-FTS input file.\n";
            if (s != "") {
                std::cout << s << std::endl;
                exit(1);
            }

            // Read simulation parameters from the input file and allocate temporary host memory for fields
            P_ = new clfts_params(inputFile);
            P_->outputParameters();
            M_=P_->M();

            // Set up random number generator
            curandCreateGenerator(&RNG_, CURAND_RNG_PSEUDO_DEFAULT);
            int seed = time(NULL);
            seed = 123456789;
            curandSetPseudoRandomGeneratorSeed(RNG_, seed);
            std::cout << "\nRNG seed: " << seed << std::endl;

            // Allocate memory for field array on the GPU and host
            w_gpu_.resize(4*M_);
            w_.resize(4*M_);

            std::cout << "\nCreating minusK_lookup() object..." << std::endl;
            minusK_lookup_ = new minusK_lookup(&(P_->m()[0]));

            std::cout << "\nCreating diblockClass() object..." << std::endl;
            dbc_ = new diblockClass(P_->NA(), P_->NB(), &(P_->m()[0]), &(P_->L()[0]), M_, &(minusK_lookup_->minusK_gpu_));

            std::cout << "\nCreating Langevin() object..." << std::endl;
            Langevin_ = new langevin_cmplx(RNG_, P_->dt(), M_);

            std::cout << "\nCreating strFuncCmplx() object..." << std::endl;
            Sk_ = new strFuncCmplx(&(P_->m()[0]), &(P_->L()[0]), M_, P_->n(), P_->XbN(), &(minusK_lookup_->minusK_gpu_));

            std::cout << "\nCreating order_parameter() object..." << std::endl;
            Psi_ = new order_parameter(&(P_->m()[0]), &(P_->L()[0]), M_, &(minusK_lookup_->minusK_gpu_));

            // Read w-[r] and w+[r] from the input file
            thrust::host_vector<thrust::complex<double>> w(2*M_);
            file_IO::readCmplxVector(w, inputFile, 2*M_, 3);

            // Copy w-(r) and w+(r) from host to GPU
            thrust::copy(w.begin(), w.end(), w_gpu_.begin());

            // Perform an initial mix to get phi-(r) and phi+(r) from the input fields
            std::cout << "\nCalculating phi-(r) and phi+(r)..." << std::endl;
            std::cout << "\nlnQ_orig = "
                      << dbc_->calc_concs(w_gpu_) 
                      << "\n\n";

            // Output initial fields
            w_ = w_gpu_;
            saveStdOutputFile("w_eq_" + std::to_string(0), &(w_[0]));
            file_IO::writeCmplxVector("phi_eq_" + std::to_string(0), &(w_[2*M_]), 2*M_);
        }



        // Destructor
        ~clfts_simulation() {
            delete dbc_;
            delete Sk_;
            delete Langevin_;
            delete Psi_;
            delete P_;
            delete minusK_lookup_;
        }



        // Equilibration loop, during which statistics are NOT sampled
        void equilibrate() {
            int it = 0;

            std::cout << output_titles() << std::endl;

            for (it=1; it<=P_->equil_its(); it++) {
                // Perform a Langevin step to update w-(r) and w+(r)
                Langevin_->step_wm_wp(w_gpu_, dbc_, RNG_, P_->sigma(), P_->XbN(), P_->zeta(), true, true, false);

                if (it % OUT_FREQ == 0) instantaneous_outputs(it, w_gpu_.begin());

                if (it % P_->save_freq() == 0) {
                    // write fields to file (copy from gpu to host)
                    w_ = w_gpu_;
                    saveStdOutputFile("w_eq_" + std::to_string(it), &(w_[0]));
                    file_IO::writeCmplxVector("phi_eq_" + std::to_string(it), &(w_[2*M_]), 2*M_);
                }

            }
            // Final save to file at end of equilibration period
            w_ = w_gpu_;
            saveStdOutputFile("w_eq_" + std::to_string(it-1), &(w_[0]));
            file_IO::writeCmplxVector("phi_eq_" + std::to_string(it-1), &(w_[2*M_]), 2*M_);
            std::cout << "\nK_ATS = " + std::to_string(Langevin_->K_ATS()) << "\n\n";
        }



        // Statistics loop, during which statistics are sampled
        void statistics() {
            thrust::host_vector<thrust::complex<double>> dQdL(3);
            int it = 0;

            std::cout << output_titles() << std::endl;

            for (it=1; it<=P_->sim_its(); it++) {
                // Perform a Langevin step to update w-(r) and w+(r)
                Langevin_->step_wm_wp(w_gpu_, dbc_, RNG_, P_->sigma(), P_->XbN(), P_->zeta(), false, true, false);

                if (it%P_->sample_freq()==0) {
                    Sk_->sample(w_gpu_);
                    dbc_->get_Q_derivatives(w_gpu_, dQdL);
                }

                if (it % OUT_FREQ == 0) instantaneous_outputs(it, w_gpu_.begin());

                if (it % P_->save_freq() == 0) {
                    // write fields to file (copy from gpu to host)
                    w_ = w_gpu_;
                    saveStdOutputFile("w_st_" + std::to_string(it), &(w_[0]));
                    file_IO::writeCmplxVector("phi_st_" + std::to_string(it), &(w_[2*M_]), 2*M_);
                    Sk_->save("Sk_" + std::to_string(it));
                }

            }
            // Final save to file at end of equilibration period
            w_ = w_gpu_;
            saveStdOutputFile("w_st_" + std::to_string(it-1), &(w_[0]));
            file_IO::writeCmplxVector("phi_st_" + std::to_string(it-1), &(w_[2*M_]), 2*M_);
            std::cout << "\n";
        }




    private:
        // Save data in a standard format to be used as in input file
        void saveStdOutputFile(std::string fileName, thrust::complex<double> *arr) {
            P_->saveOutputParams(fileName);
            file_IO::writeCmplxVector(fileName, arr, 2*M_, true);
        }

        thrust::complex<double> get_H(thrust::complex<double> lnQ, thrust::complex<double> wm_sq, thrust::complex<double> wp_sq, thrust::complex<double> wp, double XbN, double zeta) {
            return -lnQ + 0.25*XbN + wm_sq/XbN - wp_sq/(XbN + 2.0*zeta) - wp;
        }

        std::string output_titles() {
            return "it\tln(Q)\t<w-(r)>\t<w-(r)^2>\t<w+(r)>\t<w+(r)^2>\tH\tPsi";
        }

        void instantaneous_outputs(int it, thrust::device_vector<thrust::complex<double>>::iterator w_gpu_itr) {
            thrust::complex<double> lnQ, wm, wp, wm_sq, wp_sq, H, ZERO_cmplx={0.0, 0.0};
            lnQ = thrust::log(dbc_->Q());
            wm = thrust::reduce(w_gpu_itr, w_gpu_itr + M_) / M_;
            wp = thrust::reduce(w_gpu_itr + M_, w_gpu_itr + 2*M_) / M_;
            wm_sq = thrust::transform_reduce(w_gpu_itr, w_gpu_itr + M_, square_op<thrust::complex<double>>(), ZERO_cmplx, thrust::plus<thrust::complex<double>>()) / M_;
            wp_sq = thrust::transform_reduce(w_gpu_itr + M_, w_gpu_itr + 2*M_, square_op<thrust::complex<double>>(), ZERO_cmplx, thrust::plus<thrust::complex<double>>()) / M_;
            H = get_H(lnQ, wm_sq, wp_sq, wp, P_->XbN(), P_->zeta());
            std::cout   << it << "\t"
                        << lnQ << "\t"
                        << wm << "\t"
                        << wm_sq << "\t"
                        << wp << "\t"
                        << wp_sq << "\t"
                        << H << "\t"
                        << Psi_->get_Psi4(w_gpu_)
                        << std::endl;
        } 
        
};
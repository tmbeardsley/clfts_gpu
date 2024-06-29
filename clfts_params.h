// #############################################################################
// A class for dealing with the L-FTS simulation input parameters.
// Reads the input parameters from file. Not fully encapsulated as *m and 
// *L are exposed for convenience. Will act as a helper class that  
// automatically updates the values of derived parameters in future code 
// iterations (e.g., box-altering move and adaptive time step).
// #############################################################################
#pragma once
#include <stdlib.h>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <math.h>
#include <iostream>
#include <vector>


class clfts_params {

    // Input file parameters
    int    N_;                  // Total polymer length
    int    NA_;                 // Length of polymer A-block
    double XeN_;                // Effective chi multiplied by N
    double zeta_;               // Compressibility parameter
    double C_;                  // Dimensionless concentration, Nbar^0.5
    double dt_;                 // Langevin time step multiplied by N
    std::vector<int> m_;        // Number of mesh points [mx,my,mz]
    std::vector<double> L_;     // Dimensions of simulation box [Lx,Ly,Lz] in units of aN^0.5
    int    equil_its_;          // Number of Langevin steps for equilibration
    int    sim_its_;            // Number of Langevin steps for statistics
    int    sample_freq_;        // Number of Langevin steps between samples

    int    save_freq_;          // Number of steps between saving statistics to file
    //int    loadType_;           // Whether to load from file or create a new field

    // Derived parameters
    int    NB_;                 // Length of polymer B-block
    int    M_;                  // Total number of field mesh points
    double V_;                  // Volume of the simulation box
    double sigma_;              // Standard deviation of random noise multiplied by N
    double XbN_;                // Bare chi of the simulation multiplied by N

    public: 
        clfts_params(std::string inputFile) {
            m_.resize(3);
            L_.resize(3);
            read_Input_Params(inputFile);
        }

        ~clfts_params() {

        }

        // Print parameters that were read from the input file to standard output
        void outputParameters() {
            std::cout << "N = "             << N_           << std::endl;
            std::cout << "NA = "            << NA_          << std::endl;
            std::cout << "NB = "            << NB_          << std::endl;
            std::cout << "XeN = "           << XeN_         << std::endl;
            std::cout << "XbN = "           << XbN_         << std::endl;
            std::cout << "zeta = "          << zeta_        << std::endl;
            std::cout << "C = "             << C_           << std::endl;
            std::cout << "dt = "            << dt_          << std::endl;
            //std::cout << "isXeN = "         << isXeN_       << std::endl;
            std::cout << "m[0] = "          << m_[0]        << std::endl;
            std::cout << "m[1] = "          << m_[1]        << std::endl;
            std::cout << "m[2] = "          << m_[2]        << std::endl;
            std::cout << "L[0] = "          << L_[0]        << std::endl;
            std::cout << "L[1] = "          << L_[1]        << std::endl;
            std::cout << "L[2] = "          << L_[2]        << std::endl;
            std::cout << "equil_its = "     << equil_its_   << std::endl;
            std::cout << "sim_its = "       << sim_its_     << std::endl;
            std::cout << "sample_freq = "   << sample_freq_ << std::endl;
            std::cout << "save_freq_ = "    << save_freq_   << std::endl;
            //std::cout << "loadType_ = "     << loadType_    << std::endl;
            std::cout << "M_ = "            << M_           << std::endl;
            std::cout << "V_ = "            << V_           << std::endl;
            std::cout << "sigma_ = "        << sigma_       << std::endl;
        }

        // Getters
        int N() { return N_; }
        int NA() { return NA_; }
        int NB() { return NB_; }
        double XbN() { return XbN_; }
        double XeN() { return XeN_; }
        double zeta() { return zeta_; }
        double C() { return C_; }
        double dt() { return dt_; }
        //int isXeN() { return isXeN_; }
        int mx() { return m_[0]; }
        int my() { return m_[1]; }
        int mz() { return m_[2]; }
        int m(int dim) { return m_[dim]; }
        std::vector<int> m() { return m_; }
        double Lx() { return L_[0]; }
        double Ly() { return L_[1]; }
        double Lz() { return L_[2]; }
        double L(int dim) { return L_[dim]; }
        std::vector<double> L() { return L_; }
        int equil_its() { return equil_its_; }
        int sim_its() { return sim_its_; }
        int sample_freq() { return sample_freq_; }
        int save_freq() { return save_freq_; }
        //int loadType() { return loadType_; }
        int M() { return M_; }
        double V() { return V_; }
        double sigma() { return sigma_; }
        double n() { return C_*V_; }                // Total number of polymers in the system


        void saveOutputParams(std::string fileName, bool append=false) {
           //double XN_out = chi_b_;
           std::ofstream outstream;
           if (append) outstream.open(fileName,std::ios_base::app);
           else outstream.open(fileName);
           //if (isXeN_ == 1) XN_out = XeN_;
           outstream << N_ << " " << NA_ << " " << XeN_ << " " << zeta_  << " " << C_ << " " << dt_ << std::endl;
           outstream << m_[0] << " " << m_[1] << " " << m_[2] << " " << L_[0] << " " << L_[1] << " " << L_[2] << std::endl;
           outstream << equil_its_ << " " << sim_its_ << " " << sample_freq_ << " " << save_freq_ << std::endl;
           outstream.close();
        }





    private:
        // Read the simulation input parameters from file (first line)
        void read_Input_Params(std::string fileName) {
            std::string fline;
            std::ifstream instream;
            instream.open(fileName);

            // Check that the file is open
            if (!instream.is_open()) {
                std::cout << "ERROR => Couldn't open input file: " << fileName << std::endl;
                exit(1);
            }

            // Get the first line of parameters and parse
            std::getline(instream, fline);
            std::stringstream ss(fline);
            try {
                if (!(ss >> N_))        throw std::invalid_argument("Cannot read N\n");
                if (!(ss >> NA_))       throw std::invalid_argument("Cannot read NA\n");
                if (!(ss >> XeN_))      throw std::invalid_argument("Cannot read XeN\n");
                if (!(ss >> zeta_))     throw std::invalid_argument("Cannot read zeta\n");
                if (!(ss >> C_))        throw std::invalid_argument("Cannot read C\n");
                if (!(ss >> dt_))       throw std::invalid_argument("Cannot read dt\n");
            }
            //catch (invalid_argument& e) {
            catch (const std::exception& e) {
                std::cout << "ERROR => Invalid file parameters: " << e.what() << std::endl;
                exit(1);
            }

            // Get the second line of parameters and parse
            std::getline(instream, fline);
            ss.str(std::string());
            ss.clear();
            ss.str(fline);
            try {
                if (!(ss >> m_[0]))     throw std::invalid_argument("Cannot read mx\n");
                if (!(ss >> m_[1]))     throw std::invalid_argument("Cannot read my\n");
                if (!(ss >> m_[2]))     throw std::invalid_argument("Cannot read mz\n");
                if (!(ss >> L_[0]))     throw std::invalid_argument("Cannot read Lx\n");
                if (!(ss >> L_[1]))     throw std::invalid_argument("Cannot read Ly\n");
                if (!(ss >> L_[2]))     throw std::invalid_argument("Cannot read Lz\n");
            }
            //catch (invalid_argument& e) {
            catch (const std::exception& e) {
                std::cout << "ERROR => Invalid file parameters: " << e.what() << std::endl;
                exit(1);
            }

            // Get the third line of parameters and parse
            std::getline(instream, fline);
            ss.str(std::string());
            ss.clear();
            ss.str(fline);
            try {
                if (!(ss >> equil_its_))    throw std::invalid_argument("Cannot read equil_its\n");
                if (!(ss >> sim_its_))      throw std::invalid_argument("Cannot read sim_its\n");
                if (!(ss >> sample_freq_))  throw std::invalid_argument("Cannot read sample_freq\n");
                if (!(ss >> save_freq_))    throw std::invalid_argument("Cannot read save_freq\n");
            }
            //catch (invalid_argument& e) {
            catch (const std::exception& e) {
                std::cout << "ERROR => Invalid file parameters: " << e.what() << std::endl;
                exit(1);
            }


            // Redefine variables to contain and integer number of sample_freq_ periods
            equil_its_ = (equil_its_/sample_freq_)*sample_freq_;
            sim_its_ = (sim_its_/sample_freq_)*sample_freq_;  

            //// Transform from Xe to Xb if necessary
            //double z_inf = z_inf_discrete(L_, m_, N_, C_*C_);
            //if (isXeN_ == 1) {
            //    XeN_ = chi_b_;
            //    chi_b_ = chi_b_/z_inf;
            //} else {
            //    XeN_ = chi_b_*z_inf;
            //}

            // Transform from Xe to Xb
            XbN_ = XeN_ / z_inf_discrete_compressible(L_, m_, N_, C_, zeta_, 20);

            calculate_Derived_Params();
        }

        // Calculate derived parameters (should only be called after read_Input_Params())
        void calculate_Derived_Params() {
            M_ = m_[0]*m_[1]*m_[2];
            V_ = L_[0]*L_[1]*L_[2];
            sigma_ = sqrt(2.0*M_*dt_/(C_*V_));
            NB_ = N_-NA_;
        }

        double z_inf_discrete_compressible(std::vector<double>& L, std::vector<int>& m, int N, double C, double zeta, int steps = 80) {
            return z_inf_discrete(L, m, N, C * C) + dz_inf_discrete_compressible(L, m, N, C, 1.0 / zeta, 20);
            //return z_inf_discrete(L, m, N, C * C) + 0.028807;
        }

        // Calculate incompressible z_infinity (discrete chain)
        double z_inf_discrete(std::vector<double>& L, std::vector<int>& m, int N, double Nbar, int tmax=100) {
            double sum, prod=1.0, X, ell, R0, R0dL[3];
            sum = 0.5;
            R0 = sqrt((double) N);
            for (int i=0; i<3; i++) { 
                R0dL[i] = R0*L[i]/m[i];
                prod *= R0dL[i];
            }
            ell = pow(prod, 1.0/3.0);

            for (int t = 1; t <= tmax; t++) {
                X = (M_PI/ell)*sqrt(t/6.0);
                prod = 1.0;
                for (int i=0; i<3; i++) prod *= erf(X*ell/R0dL[i]);
                sum += pow(sqrt(M_PI)/(2*X), 3.0) * prod;
            }
            sum *= 2*R0/(pow(ell, 3.0)*sqrt(Nbar));
            X = (M_PI/ell)*sqrt((0.5+tmax)/6.0);

            return 1.0 - sum - 3*R0/(ell*sqrt(M_PI*Nbar)*X);
        }

        // Calculate the contribution to z_inf due to compressibility
        // Note: steps = number of steps per dimension in integration
        // Note: kappa = 1/zeta so that kappa=0 for incompressible system
        // Note: Credit for this function to Prof. Mark Matsen
        double dz_inf_discrete_compressible(std::vector<double>& L, std::vector<int>& m, int N, double C, double kappa, int steps = 80) {
            double Dz = 0.0, T, X;
            double dkx, dky, dkz, kx, ky, kz;
            int wx, wy, wz;
            if (kappa > 0.0) {
                dkx = M_PI * m[0] / (L[0] * steps);
                dky = M_PI * m[1] / (L[1] * steps);
                dkz = M_PI * m[2] / (L[2] * steps);
                for (int i = 0; i <= steps; i++) {
                    kx = i * dkx;
                    wx = (i == 0 || i == steps) ? 1 : 2;
                    for (int j = 0; j <= steps; j++) {
                        ky = j * dky;
                        wy = (j == 0 || j == steps) ? 1 : 2;
                        for (int k = 0; k <= steps; k++) {
                            kz = k * dkz;
                            wz = (k == 0 || k == steps) ? 1 : 2;
                            X = (kx*kx + ky*ky + kz*kz) / (12.0 * N);
                            T = tanh(X);
                            if (X != 0.0) Dz += wx * wy * wz * (1.0 - 1.0 / (1.0 + (T*N*kappa))) / T;
                        }
                    }
                }
                Dz *= dkx * dky * dkz / pow(2 * M_PI, 3) / (C * N);
            }
            std::cout << "Dz = " << Dz << std::endl;
            return Dz;
        }
};
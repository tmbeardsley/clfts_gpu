// #############################################################################
// Provides useful functions for file reading and writing.
// Not yet templated as current code version only has need to output doubles.
// #############################################################################
#pragma once
#include <fstream>
#include <iomanip>
#include <limits>
#include <cuda.h>

namespace file_IO {

    // Check whether a file exists
    bool isValidFile(std::string fileName) {
        if (fileName == "") return false;
        std::ifstream instream(fileName);
        return !(instream.fail());
    }

    // Read a double array from file
    template <typename T>
    void readCmplxVector(thrust::host_vector<thrust::complex<T>> &v, std::string fileName, int n, int nIgnore = 0) {
        std::ifstream instream;
        instream.open(fileName);
        // Ignore first nIgnore lines that contain parameters
        for (int i = 0; i < nIgnore; i++) instream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        // Read in the field
        T real, imag;
        for (int r = 0; r < n; r++) {
            instream >> real >> imag;
            v[r].real(real);
            v[r].imag(imag);
        }
        instream.close();
    }

    // Read a double array from file
    void readArray(double *arr, std::string fileName, int n, int nIgnore=0) {
        std::ifstream instream;
        instream.open(fileName);
        // Ignore first nIgnore lines that contain parameters
        for (int i=0; i<nIgnore; i++) instream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        // Read in the field
        for (int r=0; r<n; r++) instream >> arr[r];
        instream.close();
    }

    // Save a host array to file
    template <typename T>
    void saveArray(T *arr, std::string fileName, int n, bool append=false) {
        std::ofstream outstream;
        if (append) outstream.open(fileName,std::ios_base::app);
        else outstream.open(fileName);
        for (int r=0; r<n; r++) outstream << arr[r] << std::endl;
        outstream.close();
    }

    // // Save a host array to file
    // void saveArray(double *arr, std::string fileName, int n, bool append=false) {
    //     std::ofstream outstream;
    //     if (append) outstream.open(fileName,std::ios_base::app);
    //     else outstream.open(fileName);
    //     for (int r=0; r<n; r++) outstream << arr[r] << std::endl;
    //     outstream.close();
    // }

}
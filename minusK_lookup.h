// ##############################################################################
// Exposes a commonly used lookup table for the mapping of the 1d
// index, k (itself a mapping of the 3d reciprocal-space vector, K), 
// to the 1d index minusK[k] (a mapping of the 3d reciprocal-space vector, -K). 
// ##############################################################################

#pragma once
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class minusK_lookup {  
    
    public: 
        thrust::device_vector<int> minusK_gpu_;         // Memory to hold the minusK lookup table on the gpu

    public:
        minusK_lookup(int *m) {
            int M = m[0]*m[1]*m[2];

            // Allocate memory on host and gpu
            thrust::host_vector<int> minusK_host(M);
            minusK_gpu_.resize(M);

            calc_minusK(&(minusK_host[0]), m);
            minusK_gpu_ = minusK_host;
        }

    private:
        // Calculate the wavevector moduli and store in K[]
        void calc_minusK(int *minusK, int *m) {
            int K0, K1, K2, k;
            int mK0, mK1, mK2;

            for (int k0=-(m[0]-1)/2; k0<=m[0]/2; k0++) {
                K0 = (k0<0)?(k0+m[0]):k0;
                mK0 = (k0>0)?(-k0+m[0]):-k0;

                for (int k1=-(m[1]-1)/2; k1<=m[1]/2; k1++) {
                    K1 = (k1<0)?(k1+m[1]):k1;
                    mK1 = (k1>0)?(-k1+m[1]):-k1;

                    for (int k2=-(m[2]-1)/2; k2<=m[2]/2; k2++) {
                        K2 = (k2<0)?(k2+m[2]):k2;
                        mK2 = (k2>0)?(-k2+m[2]):-k2;

                        k = K2+m[2]*(K1+m[1]*K0);
                        minusK[k] = mK2+m[2]*(mK1+m[1]*mK0);
                    }
                }
            }
        }

};

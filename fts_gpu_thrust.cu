// fts.cu
//------------------------------------------------------------
// GPU version of the FTS code for a diblock copolymer melt
// Note that lengths are expressed in units of R0=a*N^0.5
//------------------------------------------------------------

#include "clfts_simulation.h"


//------------------------------------------------------------
int main()
{
    clfts_simulation *clfts = new clfts_simulation("input");

    clfts->equilibrate();

    clfts->statistics();

    delete clfts;


}

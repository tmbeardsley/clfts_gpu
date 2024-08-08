// fts.cu
//------------------------------------------------------------
// GPU version of the FTS code for a diblock copolymer melt
// Note that lengths are expressed in units of R0=a*N^0.5
//------------------------------------------------------------

#include "clfts_simulation.h"
#include <string>

//------------------------------------------------------------
int main(int argc, char *argv[])
{
    // Get input file name from command-line argument
    if (argc != 2) {
        std::cout << "Please supply a single input file name as a command-line argument.\n\n";
        exit(1);
    }
    std::string inputFile(argv[1]);

    // Create a new complex langevin simulation
    clfts_simulation *clfts = new clfts_simulation(inputFile);

    // Run the simulation to reach equilibrium
    clfts->equilibrate();

    // Start running from the equilibrated system to gather statistics
    clfts->statistics();

    // Delete the clfts_simulation instance
    delete clfts;
}

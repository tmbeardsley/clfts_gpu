# Complex Langevin Field-Theoretic Simulation of Diblock Copolymers on the GPU

## Description
See https://www.tbeardsley.com/projects/lfts/clfts_gpu for a detailed discussion of this project.<br>

## Required Dependencies
CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit/)<br>

## Compiling
Two methods of compiling the program are available:<br>
<ol>
  <li><b>comp.sh</b>
    <br>
    A simple bash script to create a 'build' directory containing the compiled program code: clfts-gpu.<br><br>
    On a Linux system, run the bash script from the top directory via:<br>
    <b>sh comp.sh</b>
    <br><br>
  </li>
  <li><b>CMake</b>
    <br>
    CMakeLists.txt specifies the required commands for CMake to create (and run) Makefiles, which create a 'build' directory and compile the program code as: clfts-gpu.<br><br>
    From the top directory, run: <br>
    <b>cmake -B build</b><br>
    <b>cmake --build build</b>
  </li>
</ol>

## Running the program
After compilation the executable file, clfts-gpu, resides in the 'build' directory. An input file must be supplied to the executable at the command line, examples of which are contained in the 'input_files' folder. 
For example, from the top level of the directory tree, the program could be run via: <br><br>
<b>./build/clfts-gpu ./input_files/input</b>

## Input Files
The input_files directory contains example input files that can be supplied to the program from the command line.

### Input file format
Line 1: <em>N NA XeN zetaN C Ndt</em><br>
Line 2: <em>mx my mz Lx Ly Lz</em><br>
Line 3: <em>n_eq n_st n_smpl save_freq</em><br>
Lines 4->(M+3): REAL[W-(r)] IMAG[W-(r)]<br>
Lines (M+4)->(2M+3): REAL[W+(r)] IMAG[W+(r)]<br>

<b>Notes:</b><br>
A real-space position r = (x,y,z) corresponds to a mesh point position r_m = (i,j,k), where i=0->mx-1, j=0->my-1 and k=0->mz-1 are integers. The elements of the fields, W-(r) and W+(r), are then written in ascending order of the row-major index: p = mx\*(i\*my+j)+k.<br>

#### Parameter descriptions
| Parameter | Type | Description |
| :---: | :---: | --- |
| <em>N</em> | Integer | Number of monomers in a single polymer chain (integer) |
| <em>NA</em> | Integer | Number of monomers in the A-block of a polymer chain (integer) |
| <em>XeN</em> | Integer | Interaction strength between A and B-type monomers (double) |
| <em>zetaN</em> | Integer | Compressibility factor, zeta, multiplied by N (double) |
| <em>C</em> | Integer | Square root of the invariant polymerisation index, Nbar (double) |
| <em>Ndt</em> | Integer | Size of the time step in the Langevin update of W-(r) (double) |
| <em>mx, my, mz</em> | Integer | Number of mesh points in the x, y, and z dimensions of the simulation box (integers) |
| <em>Lx, Ly, Lz</em> | Integer | Dimensions of the simulation box (in units of the polymer end-to-end length, R0) in the x, y, and z dimensions (doubles) |
| <em>n_eq</em> | Integer | Number of langevin steps performed to equilibrate the system (integer) |
| <em>n_st</em> | Integer | Number of langevin steps performed after equilibration has ended, during which statistics are sampled (integer) |
| <em>n_smpl</em> | Integer | Number of steps between samples being taken in the statistics period (integer) |
| <em>save_freq</em> | Integer | Number of steps between saving outputs to file |
| M | Integer | Total number of mesh points (M = mx\*my\*mz), such that the proceeding 2*M lines of the file can hold W-(r) and w+(r) fields to load |

## Output files
#### w_eq_<step_number>
The state of the W-(r) and w+(r) fields at simulation step number <step_number> during the equilibration period. First three lines are simulation parameters so it can be used as an input file.<br>

#### w_st_<step_number>
The state of the W-(r) and w+(r) fields at simulation step number <step_number> during the statistics gathering period. First three lines are simulation parameters so it can be used as an input file.<br>

#### phi_eq_<step_number>
The state of the phi-(r) and phi+(r) fields at simulation step number <step_number> during the equilibration period.<br>

#### phi_st_<step_number>
The state of the phi-(r) and phi+(r) fields at simulation step number <step_number> during the statistics gathering period.<br>

#### Sk_<step_number>
The spherically-averaged structure function at simulation step number <step_number> during the statistics gathering period.<br>


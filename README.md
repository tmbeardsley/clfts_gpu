# Complex Langevin Field-Theoretic Simulation of Diblock Copolymers on the GPU


| <img src="https://www.tbeardsley.com/imgs/projects/lfts/clfts_gpu/clfts.png" width="600px"> |
| :---: |
| Real and imaginary parts of W<sub>-</sub>(**r**) at Nbar=10<sup>5</sup>, N=90, &chi;N=12, m=32 and L=4.209. |



## 1. Description
See https://www.tbeardsley.com/projects/lfts/clfts_gpu for a detailed discussion of this project.<br>

## 2. Required Dependencies
CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit/)<br>

## 3. Compiling
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

## 4. Running the program
After compilation the executable file, clfts-gpu, resides in the 'build' directory. An input file must be supplied to the executable at the command line, examples of which are contained in the 'input_files' folder. 
For example, from the top level of the directory tree, the program could be run via: <br><br>
<b>./build/clfts-gpu ./input_files/input</b>

## 5. Input Files
The input_files directory contains example input files that can be supplied to the program from the command line.

### 5a. Input file format
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
| <em>N</em> | Integer | Number of monomers in a single polymer chain |
| <em>NA</em> | Integer | Number of monomers in the A-block of a polymer chain |
| <em>XeN</em> | Double | Interaction strength between A and B-type monomers |
| <em>zetaN</em> | Double | Compressibility factor, zeta, multiplied by N |
| <em>C</em> | Double | Square root of the invariant polymerisation index, Nbar |
| <em>Ndt</em> | Double | Size of the time step in the Langevin update of W-(r) |
| <em>mx, my, mz</em> | Integers | Number of mesh points in the x, y, and z dimensions of the simulation box |
| <em>Lx, Ly, Lz</em> | Doubles | Dimensions of the simulation box (in units of the polymer end-to-end length, R0) in the x, y, and z dimensions |
| <em>n_eq</em> | Integer | Number of langevin steps performed to equilibrate the system |
| <em>n_st</em> | Integer | Number of langevin steps performed after equilibration has ended, during which statistics are sampled |
| <em>n_smpl</em> | Integer | Number of steps between samples being taken in the statistics period |
| <em>save_freq</em> | Integer | Number of steps between saving outputs to file |
| M | Integer | Total number of mesh points (M = mx\*my\*mz), such that the proceeding 2*M lines of the file can hold the W-(r) and w+(r) fields to be loaded |

## 6. Output files
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

## 7. Visualisation Script
The tools folder in the root directory contains a simple script for taking the first <em>M</em> lines of a w_<..>_<step_number> or phi_<..>_<step_number> output file from the simulation, and creating a .vtk file that can be loaded into <a href="https://www.paraview.org" target="_blank">Paraview</a> for visualisation as a volume plot. Note that the first <em>M</em> lines are used as they correspond to the W-(r) or phi-(r) fields, which are usually the ones of interest. The script could easily be edited to use lines <em>M</em>+1 to 2<em>M</em> in order to plot w+(r) or phi+(r) instead.

### 7a. How to use make_vtk.sh
The script can be run from the command line as follows:<br><br>
<b>sh make_vtk.sh \<path_to_file_to_visualise\> \<mx\> \<my\> \<mz\></b>
<br><br>
where <em>mx</em>, <em>my</em> and <em>mz</em> are the number of grid points in the x, y and z-dimensions of the file being visualised. 
The script's output file name will be the same as \<path_to_file_to_visualise\>, but with a .vtk extension.

#!/bin/bash
#$ -cwd

mx=40
my=40
mz=40

dx=0.1
dy=0.1
dz=0.1

((M=mx*my*mz))

echo 'mx='$mx
echo 'my='$my
echo 'mz='$mz
echo 'M='$M
echo 'dx='$dx
echo 'dy='$dy
echo 'dz='$dz

in_file=${1}
out_file=${1}.vtk
echo '# vtk DataFile Version 2.0' > ./${out_file}
echo "CT Density" >> ./${out_file}
echo "ASCII" >> ./${out_file}
echo >> ./${out_file}
echo "DATASET STRUCTURED_POINTS" >> ./${out_file}
echo "DIMENSIONS ${mx} ${my} ${mz}" >> ./${out_file}
echo "ORIGIN 0.000000 0.000000 0.000000" >> ./${out_file}
echo "SPACING ${dx} ${dy} ${dz}" >> ./${out_file}
echo >> ./${out_file}
echo "POINT_DATA ${M}" >> ./${out_file}
echo "SCALARS scalars float" >> ./${out_file}
echo "LOOKUP_TABLE default" >> ./${out_file}
echo >> ./${out_file}

awk -F '\t' '{print $1}' ${in_file} >> ./${out_file}
#echo ${line} >> ./${out_file}


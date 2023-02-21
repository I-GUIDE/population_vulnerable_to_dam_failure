#!/bin/bash

for i in {0..14} 
do
    sbatch h4_sbatch.sbatch $i
    echo $i "Regular run"
done

# for i in {0..16} 
# do
#     if [ $i -lt 17 ] 
#     then 
#         sbatch h4_sbatch.sbatch $i
#         echo $i "Regular run"
#     else
#         sbatch h24_sbatch.sbatch $i
#         echo $i "Intense run"
#     fi
# done
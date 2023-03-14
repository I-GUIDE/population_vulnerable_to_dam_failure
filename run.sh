#!/bin/bash

# for i in {0..14} 
# do
#     sbatch h4_sbatch.sbatch $i
#     echo $i "Regular run"
# done

for i in {0..17} 
do
    if [ $i -lt 13 ] 
    then 
        sbatch h4_sbatch.sbatch $i
        echo $i "Regular run"
    else
        sbatch h12_sbatch.sbatch $i
        echo $i "Intense run"
    fi
done
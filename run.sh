#!/bin/bash

for i in {0..18} 
do
echo $i
    if [ $i -lt 17 ] 
    then 
        sbatch h4_combined_process.sbatch $i
        echo $i "Regular run"
    else
        sbatch h24_combined_process.sbatch $i
        echo $i "Intense run"
    fi
done
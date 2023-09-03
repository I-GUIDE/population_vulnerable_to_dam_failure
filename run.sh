#!/bin/bash

for i in {0..17} 
do
    if [ $i -lt 15 ] 
    then 
        sbatch h1_sbatch.sbatch $i
        echo $i "Regular run"
    else
        sbatch h6_sbatch.sbatch $i
        echo $i "Intense run"
    fi
done
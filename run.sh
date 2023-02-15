#!/bin/bash

for i in {1..28}
do
    sbatch combined_process_no_benchmark.sbatch i
done
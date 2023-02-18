#!/bin/tcsh
#SBATCH --job-name=dam
#SBATCH --partition=wholenode
#SBATCH --nodes=1

#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=jparkgeo@illinois.edu

module purge
module load gcc
module load openmpi

set var=$1
singularity exec --bind /anvil/scratch/x-cybergis/compute/1672794633Ilt8J:/job/executable /home/x-cybergis/simages/extractinundationcensustracts-processor.sif python3 combined_process_no_benchmark_copy.py $var

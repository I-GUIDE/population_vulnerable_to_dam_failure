#!/bin/tcsh
#SBATCH --job-name=dam
#SBATCH --partition=shared
#SBATCH --nodes=1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=24000MB
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=jparkgeo@illinois.edu

module purge
module load gcc
module load openmpi

set var=$1
singularity exec --bind /anvil/scratch/x-cybergis/compute/1672794633Ilt8J:/job/executable /home/x-cybergis/simages/extractinundationcensustracts-processor.sif python3 combined_process_no_benchmark.py $var

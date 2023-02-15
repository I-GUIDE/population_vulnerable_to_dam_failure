#!/bin/tcsh
#SBATCH --job-name=dam
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2000MB
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=jparkgeo@illinois.edu

module purge
module load gcc
module load openmpi

singularity exec --bind /anvil/scratch/x-cybergis/compute/1672794633Ilt8J:/job/executable /home/x-cybergis/simages/extractinundationcensustracts-processor.sif python3 dam_download_batch.py
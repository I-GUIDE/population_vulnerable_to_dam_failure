#!/bin/tcsh
#SBATCH --job-name=dam
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16GB
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --mail-user=jparkgeo@illinois.edu

module purge
cd /data/cigi/common/jparkgeo/population_vulnerable_to_dam_failure
source /data/keeling/a/jparkgeo/.bashrc

set var=$1
conda run -n aging_dam python combined_process_no_benchmark.py $var
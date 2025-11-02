#!/bin/bash -l

# SBATCH header (scheduler-side resource request)
#SBATCH --job-name=knn-test
#SBATCH -p gpu                     # use the gpu partition
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1          # Number of CPU threads per MPI rank
#SBATCH --ntasks=4                 # default of 4 MPI processes
#SBATCH --time=00:30:00
#SBATCH --output test-run-%j.out

set -euo pipefail
module purge
module load mpi cuda

ITERATIONS=10
N="${SLURM_NTASKS:-1}"

export I_MPI_HYDRA_BOOTSTRAP=slurm
HYDRA_BOOTSTRAP_ARG="-bootstrap slurm"

echo "Running MPI/CUDA on ${SLURM_JOB_NUM_NODES:-?} node(s), ${N} ranks, ${SLURM_CPUS_PER_TASK:-1} CPU/rank, ${SLURM_GPUS_PER_TASK:-?} GPU/rank"
for size in small medium large; do
  train="datasets/${size}-train.arff"
  test="datasets/${size}-test.arff"
  if [ -f datasets/${size}-train.arff ] && [ -f datasets/${size}-test.arff ]; then
    for i in $(seq 1 "${ITERATIONS}"); do
      mpirun ${HYDRA_BOOTSTRAP_ARG} -np "${N}" ./cuda-mpi "${train}" "${test}" 3
    done
  else
    echo "mpi: Dataset files for size ${size} not found, skipping."
  fi
done


#!/bin/bash -l

# SBATCH header (scheduler-side resource request)
#SBATCH --job-name=cuda-knn-test
#SBATCH -p gpu                     # use the gpu partition
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1          # Number of CPU threads per MPI rank
#SBATCH --time=00:30:00
#SBATCH --output cuda-run-%j.out

set -euo pipefail
module purge
module load cuda

ITERATIONS=10

echo "Running CUDA version"
for size in small medium large; do
  train="datasets/${size}-train.arff"
  test="datasets/${size}-test.arff"
  if [ -f datasets/${size}-train.arff ] && [ -f datasets/${size}-test.arff ]; then
    for i in $(seq 1 "${ITERATIONS}"); do
      ./cuda "${train}" "${test}" 3
    done
  else
    echo "Dataset files for size ${size} not found, skipping."
  fi
done


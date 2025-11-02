#!/bin/bash -l

# SBATCH header (scheduler-side resource request)
#SBATCH --job-name=serial-knn-test
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output serial-run-%j.out

set -euo pipefail

ITERATIONS=10

echo "Running Serial version"
for size in small medium large; do
  train="datasets/${size}-train.arff"
  test="datasets/${size}-test.arff"
  if [ -f datasets/${size}-train.arff ] && [ -f datasets/${size}-test.arff ]; then
    for i in $(seq 1 "${ITERATIONS}"); do
      ./serial "${train}" "${test}" 3
    done
  else
    echo "Dataset files for size ${size} not found, skipping."
  fi
done


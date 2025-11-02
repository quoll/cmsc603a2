# cmsc603a2
A cuda assignment.

Note that a lot of the code has been provided in a class, and is not how I would like to write it.

In my first assignment for this subject I changed all the calls to `malloc` into `new`, used vectors, safe pointers, and more. But this was hard for the TAs to grade, so we were asked to stick to the basic C-style constructs provided to us. Hence, this supposedly C++ project looks like a cross between something in the early 90s with the CUDA libraries of the last decade. Not my preferred style, but it is built to explore CUDA, and not exemplify modern code.

While not useful on the example datasets, there was also a request to expand this code to use MPI. The coordination of the data across multiple processes has more overhead than is useful for the provided datasets, but the principles are demonstrated.

## File Descriptions
The following source files all do kNN in slightly different ways:
* serial.cpp - A C++ file that does kNN as a nested loop, where the outer loop is every item to be tested, and the inner loop is each item of training data to compare the test item against. Built using `g++`.
* cuda.cu - CUDA/C++ to perform kNN where each item to be tested gets its own thread. Built using `nvcc`.
* cuda-single.cu - CUDA/C++ to simulate the `cuda.cu` program, but to restrict running to a single thread that uses an outer loop like `serial.cpp` does. This is a terrible idea, but it is to gain a sense of the speed difference between a CPU and a GPU SM. Built using `nvcc`.
* cuda-mpi.cu - CUDA/MPI/C++ to break up the test data into multiple processes (via MPI), for each process to execute the same CUDA kernel as `cuda.cu` on its group of data, then to return the results for each group via MPI. Built using `nvcc` with an additional flag of `-ccbin=mpicxx`.

The build files:
* Makefile - A Makefile for building all of the above. The valid targets are: `all`, `clean`, `serial`, `cuda`, `cuda-single`, `cuda-mpi`.
* makea2 - A helper script for running `make`. This is built on a cluster with GPUs, but the central host does not have a GPU, nor does it have the appropriate compiler and libraries. This script handles connection to a node with a GPU, initializes CUDA and MPI, changes to the assignment directory, and executes `make` on the requested target. If no target it provided, then it calls defaults to `make all`.

The GPU cluster is managed via SLURM, and running requires a job configuration that is appropriate to the task. Each of the SLURM scripts can be exeuted via `sbatch`:
* mpi\_cuda.slurm - A script for executing `cuda-mpi` 10 times on each dataset (small, medium, large) on a GPU node, requesting a number of processes to do so (default: 4). The number of processes can be overridden by passing `--ntasks=`_**N**_ to `sbatch`.
* cuda-single.slurm - A script for executing `cuda-single` 10 times on each dataset (small, medium, large) on a GPU node.
* cuda.slurm - A script for executing `cuda` 10 times on each dataset (small, medium, large) on a GPU node.
* serial.slurm - A script for executing `serial` 10 times on each dataset (small, medium, large) on any compute node.

## Use
Assuming a default directory of `dev/a2` (this is hard coded in `makea2`, but can be changed), then after connecting to the central host, the code can be built and then any of the slurm scripts can be scheduled. For instance, mpi\_cuda on 4 processes, overriding the job name and output file:
```bash
$ cd dev/a2
$ ./makea2
$ sbatch --ntasks=8 --output=mpi-8-%j.out --job-name=mpi8 mpi_cuda.slurm
```


# cmsc603a2
A cuda assignment.

Note that a lot of the code has been provided in a class, and is not how I would like to write it.

In my first assignment for this subject I changed all the calls to `malloc` into `new`, used vectors, safe pointers, and more. But this was hard for the TAs to grade, so we were asked to stick to the basic C-style constructs provided to us. Hence, this supposedly C++ project looks like a cross between something in the early 90s with the CUDA libraries of the last decade. Not my preferred style, but it is built to explore CUDA, and not exemplify modern code.

While not useful on the example datasets, there was also a request to expand this code to use MPI. The coordination of the data across multiple processes has more overhead than is useful for the provided datasets, but the principles are demonstrated.


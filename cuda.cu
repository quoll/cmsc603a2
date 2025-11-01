#include <stdio.h>
#include <sys/errno.h>
#include <cfloat>
#include <cuda_runtime.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

__device__ float distanceSq(float* instance_A, float* instance_B, int num_attributes) {
    float sum = 0;

    for (int i = 0; i < num_attributes-1; i++) {
        float diff = instance_A[i] - instance_B[i];
        sum += diff*diff;
    }

    return sum;
}

__global__ void kNN(int* predictions,  // the result
                    float* test_matrix, float* train_matrix,          // data matrices
                    int test_num_instances, int train_num_instances,  // limits of data matrices
                    int num_attributes,                               // dimensions
                    int num_classes, int k) {                         // parameters to operate within

  extern __shared__ unsigned char smem[];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // use 1 thread per test entry
  if (tid < test_num_instances) {
    float* sharedCandidates = (float*)smem;
    float* candidates = sharedCandidates + (threadIdx.x * 2 * k);

    int* sharedClassCounts = (int*)(smem + (blockDim.x * 2 * k * sizeof(float)));
    int* classCounts = sharedClassCounts + threadIdx.x * num_classes;

    for (int i = 0; i < k * 2; i++) candidates[i] = FLT_MAX;
    for (int i = 0; i < num_classes; i++) classCounts[i] = 0;

    // this thread searches the entire train list to get the top k
    int testBase = tid * num_attributes;
    for (int keyIndex = 0; keyIndex < train_num_instances; keyIndex++) {
      int trainBase = keyIndex * num_attributes;
      float distSq = distanceSq(&test_matrix[testBase], &train_matrix[trainBase], num_attributes);
      for (int c = 0; c < k; c++){
        if (distSq < candidates[2 * c]) {
          // Found a new candidate
          // Shift previous candidates down by one
          for (int x = k - 2; x >= c; x--) {
            candidates[2 * x + 2] = candidates[2 * x];
            candidates[2 * x + 3] = candidates[2 * x + 1];
          }

          // Set key vector as potential k NN
          candidates[2 * c] = distSq;
          candidates[2 * c + 1] = train_matrix[trainBase + num_attributes - 1]; // class value
          break;
        }
      }
    }

    // Bincount the candidate labels and pick the most common
    for (int i = 0; i < k; i++) {
      classCounts[(int)candidates[2 * i + 1]]++;
    }

    int max_value = -1;
    int max_class = 0;
    for (int i = 0; i < num_classes; i++) {
      if (classCounts[i] > max_value) {
        max_value = classCounts[i];
        max_class = i;
      }
    }

    predictions[tid] = max_class;
  }
}

int* computeConfusionMatrix(int* predictions, ArffData* dataset) {
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matrix size numberClasses x numberClasses

    for(int i = 0; i < dataset->num_instances(); i++) { // for each instance compare the true class and predicted class
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();
        int predictedClass = predictions[i];
        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }
    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset) {
    int successfulPredictions = 0;
    for(int i = 0; i < dataset->num_classes(); i++) {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }
    return 100 * successfulPredictions / (float) dataset->num_instances();
}

// Queries the device for the maximum shared memory available in a block.
// We know this is large enough (over 100kb) but it's good to check
size_t maxCudaSharedMemory() {
  int device;
  cudaGetDevice(&device);
  int maxDefault = 0;
  cudaDeviceGetAttribute(&maxDefault, cudaDevAttrMaxSharedMemoryPerBlock, device);
  return (size_t)maxDefault;
}

// Perform the Cuda setup and calling. Requires a result array that will be filled in.
// Returns a new predictions array that must be freed.
// Updates the milliseconds pointer with the time for execution.
int* hostKNN(float* h_test_matrix, float* h_train_matrix,      // data matrices
             int test_num_instances, int train_num_instances,  // limits of data matrices
             int num_attributes, int num_classes, int k,       // dimensions
             float* pMilliseconds) {

  // allocate memory on the device
  float *d_test_matrix, *d_train_matrix;
  int testElements = test_num_instances * num_attributes;
  int trainElements = train_num_instances * num_attributes;
  cudaMalloc(&d_test_matrix, testElements * sizeof(float));
  cudaMalloc(&d_train_matrix, trainElements * sizeof(float));

  int *d_predictions;
  cudaMalloc(&d_predictions, test_num_instances * sizeof(int));

  // copy data to the device
  cudaMemcpy(d_test_matrix, h_test_matrix, testElements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_train_matrix, h_train_matrix, trainElements * sizeof(float), cudaMemcpyHostToDevice);

  // determine the amount of shared memory available per block
  // set up timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 1 thread per test. Group into blocks of 256
  int threadsPerBlock = 256;
  int blocksPerGrid = (test_num_instances + threadsPerBlock - 1) / threadsPerBlock;
  size_t sharedMemorySize = threadsPerBlock * (2 * k * sizeof(float) + num_classes * sizeof(int));
  size_t maxShm = maxCudaSharedMemory();
  if (sharedMemorySize > maxShm) {
    printf("Requires too much shared memory per block. Required = %zu. Available = %zu\n", sharedMemorySize, maxShm);
    exit(2);
  }
  printf("%d blocks of %d threads.\n", blocksPerGrid, threadsPerBlock);

  // start timer
  cudaEventRecord(start);

  kNN<<<blocksPerGrid, threadsPerBlock, sharedMemorySize>>>(d_predictions, d_test_matrix, d_train_matrix, test_num_instances, train_num_instances,
                                                            num_attributes, num_classes, k);

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    printf("CUDA Error on launching the kernel: %s\n", cudaGetErrorString(err));
    exit(3);
  }
  // end timer
  cudaEventRecord(stop);
  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess) {
    printf("CUDA Error during kernel execution: %s\n", cudaGetErrorString(err));
    exit(4);
  }
  if (pMilliseconds != nullptr) {
    cudaEventElapsedTime(pMilliseconds, start, stop);
  }

  // allocate host memory for the results and retrieve them from the device
  int* h_predictions = (int*)malloc(test_num_instances * sizeof(int));
  cudaMemcpy(h_predictions, d_predictions, test_num_instances * sizeof(int), cudaMemcpyDeviceToHost);

  // clean up the device resources
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_test_matrix);
  cudaFree(d_train_matrix);
  cudaFree(d_predictions);

  // Return the results
  return h_predictions;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
      printf("Usage: %s datasets/train.arff datasets/test.arff k\n", argv[0]);
      exit(0);
  }

  // k value for the k-nearest neighbors
  errno = 0;
  int k = strtol(argv[3], NULL, 10);
  if (errno != 0) {
    printf("k value must be an integer. Got '%s'\n", argv[3]);
    exit(1);
  }

  // Open the datasets
  ArffParser parserTest(argv[2]);
  ArffParser parserTrain(argv[1]);
  ArffData *test = parserTest.parse();   // owned and freed by parserTest
  ArffData *train = parserTrain.parse(); // owned and freed by parserTrain

  int test_num_instances = test->num_instances();
  int train_num_instances = train->num_instances();

  float milliseconds = 0.0;
  int* predictions = hostKNN(test->get_dataset_matrix(), train->get_dataset_matrix(),
                             test_num_instances, train_num_instances,
                             train->num_attributes(), train->num_classes(), k, &milliseconds);

  // Compute the confusion matrix
  int* confusionMatrix = computeConfusionMatrix(predictions, test);
  // Calculate the accuracy
  float accuracy = computeAccuracy(confusionMatrix, test);

  printf("The %i-NN classifier for %d test instances and %d train instances required %f ms GPU time for GPU. Accuracy was %.2f%%\n",
           k, test_num_instances, train_num_instances, milliseconds, accuracy);

  free(confusionMatrix);
  free(predictions);

  return 0;
}

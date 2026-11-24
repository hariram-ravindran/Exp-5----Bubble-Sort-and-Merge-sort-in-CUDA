# Exp5 Bubble Sort and Merge sort in CUDA
**Objective:**
Implement Bubble Sort and Merge Sort on the GPU using CUDA, analyze the efficiency of this sorting algorithm when parallelized, and explore the limitations of Bubble Sort and Merge Sort for large datasets.
## AIM:
Implement Bubble Sort and Merge Sort on the GPU using CUDA to enhance the performance of sorting tasks by parallelizing comparisons and swaps within the sorting algorithm.

Code Overview:
You will work with the provided CUDA implementation of Bubble Sort and Merge Sort. The code initializes an unsorted array, applies the Bubble Sort, Merge Sort algorithm in parallel on the GPU, and returns the sorted array as output.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC, Google Colab with NVCC Compiler, CUDA Toolkit installed, and sample datasets for testing.

## PROCEDURE:

Tasks:

a. Modify the Kernel:

Implement Bubble Sort and Merge Sort using CUDA by assigning each comparison and swap task to individual threads.
Ensure the kernel checks boundaries to avoid out-of-bounds access, particularly for edge cases.
b. Performance Analysis:

Measure the execution time of the CUDA Bubble Sort with different array sizes (e.g., 512, 1024, 2048 elements).
Experiment with various block sizes (e.g., 16, 32, 64 threads per block) to analyze their effect on execution time and efficiency.
c. Comparison:

Compare the performance of the CUDA-based Bubble Sort and Merge Sort with a CPU-based Bubble Sort and Merge Sort implementation.
Discuss the differences in execution time and explain the limitations of Bubble Sort and Merge Sort when parallelized on the GPU.
## PROGRAM:
```
%%writefile sorting.cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>

// GPU Bubble Sort Kernel
__global__ void bubbleSortKernel(int *d_arr, int n)
{
    int idx = threadIdx.x;

    for (int i = 0; i < n - 1; i++)
    {
        if (idx < n - i - 1)
        {
            if (d_arr[idx] > d_arr[idx + 1])
            {
                int temp = d_arr[idx];
                d_arr[idx] = d_arr[idx + 1];
                d_arr[idx + 1] = temp;
            }
        }
        __syncthreads();
    }
}

// Device merge function
__device__ void merge(int *arr, int left, int mid, int right)
{
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int L[1024];
    int R[1024];

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];

    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];

    while (j < n2)
        arr[k++] = R[j++];
}

// GPU Merge Sort Kernel
__global__ void mergeSortKernel(int *d_arr, int *d_temp, int n)
{
    if (threadIdx.x == 0)
    {
        for (int size = 1; size < n; size *= 2)
        {
            int left = 0;

            while (left + size < n)
            {
                int mid = left + size - 1;
                int right = min(left + 2 * size - 1, n - 1);

                merge(d_arr, left, mid, right);
                left += 2 * size;
            }

            for (int i = 0; i < n; i++)
                d_temp[i] = d_arr[i];

            for (int i = 0; i < n; i++)
                d_arr[i] = d_temp[i];
        }
    }
}

// CPU Merge helper
void mergeHost(int *arr, int left, int mid, int right)
{
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int*)malloc(n1 * sizeof(int));
    int *R = (int*)malloc(n2 * sizeof(int));

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];

    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2)
    {
        if (L[i] <= R[j])
            arr[k++] = L[i++];
        else
            arr[k++] = R[j++];
    }

    while (i < n1)
        arr[k++] = L[i++];

    while (j < n2)
        arr[k++] = R[j++];

    free(L);
    free(R);
}

// GPU Bubble Sort
void bubbleSort(int *arr, int n)
{
    int *d_arr;

    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    bubbleSortKernel<<<1, n>>>(d_arr, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Bubble Sort (GPU) took %f milliseconds\n", milliseconds);
}

// GPU Merge Sort
void mergeSort(int *arr, int n)
{
    int *d_arr, *d_temp;

    cudaMalloc((void**)&d_arr, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));

    cudaMemcpy(d_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    mergeSortKernel<<<1, 1>>>(d_arr, d_temp, n);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_temp);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Merge Sort (GPU) took %f milliseconds\n", milliseconds);
}

// CPU Bubble Sort
void bubbleSortCPU(int *arr, int n)
{
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n - 1; i++)
    {
        for (int j = 0; j < n - i - 1; j++)
        {
            if (arr[j] > arr[j + 1])
            {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Bubble Sort (CPU) took %f milliseconds\n", duration.count());
}

// CPU Merge Sort
void mergeSortCPU(int *arr, int n)
{
    auto start = std::chrono::high_resolution_clock::now();

    for (int size = 1; size < n; size *= 2)
    {
        int left = 0;

        while (left + size < n)
        {
            int mid = left + size - 1;
            int right = std::min(left + 2 * size - 1, n - 1);

            mergeHost(arr, left, mid, right);
            left += 2 * size;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Merge Sort (CPU) took %f milliseconds\n", duration.count());
}

// Print array
void printArray(int *arr, int n)
{
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);

    printf("\n");
}

// Main
int main()
{
    int n = 10000;
    int *arr = (int*)malloc(n * sizeof(int));

    // Generate random array
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    printf("Original array: \n");
    printArray(arr, n);

    // CPU Bubble Sort
    bubbleSortCPU(arr, n);
    printf("Sorted array using Bubble Sort (CPU): \n");
    printArray(arr, n);

    // Generate random array again
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    // GPU Bubble Sort
    bubbleSort(arr, n);
    printf("Sorted array using Bubble Sort (GPU): \n");
    printArray(arr, n);

    // Generate random array again
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    printf("Original array: \n");
    printArray(arr, n);

    // CPU Merge Sort
    mergeSortCPU(arr, n);
    printf("Sorted array using Merge Sort (CPU): \n");
    printArray(arr, n);

    // Generate random array again
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;

    // GPU Merge Sort
    mergeSort(arr, n);
    printf("Sorted array using Merge Sort (GPU): \n");
    printArray(arr, n);

    free(arr);
    return 0;
}
```
## OUTPUT:
<img width="1082" height="380" alt="image" src="https://github.com/user-attachments/assets/9f7d6c03-72ac-4d56-9340-36dcb298aa20" />


## RESULT:
Thus, the program has been executed using CUDA to compare CPU and GPU implementations of Bubble Sort and Merge Sort. It is observed that CPU Bubble Sort took 331.743226 nanoseconds (331.74 milliseconds), GPU Bubble Sort took 165.333725 milliseconds, CPU Merge Sort took 2.114585 nanoseconds (2.11 milliseconds), and GPU Merge Sort took 340.994568 milliseconds. Hence, for this dataset, GPU Bubble Sort performed faster than CPU Bubble Sort, while CPU Merge Sort performed faster than GPU Merge Sort..

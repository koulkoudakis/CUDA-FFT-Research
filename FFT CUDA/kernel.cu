/*
Author: Sharome Burton
Date: 20/June/2020
Title: CUDA Parallel Implementation of Radix 2 FFT

This program runs from kernel.cu.
This program executes the Radix-2 Fast Fourier Transform on two 32768-element arrays
multiple times using GPU parallel processing capabilities. The time taken to execute the main function is measured using the clock()
function
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


cudaError_t CudaFFT(const float *RX, const float *IX, float *RFFT, float *IFFT, unsigned int size);

__global__ void FFT(const float *RX, const float *IX, float *RFFT, float *IFFT, int size)
{
	const float PI = 3.14159;
	float numElements = 4;
	int i = numElements * (blockDim.x * blockIdx.x +  threadIdx.x);

	if (i < size) 
	{
		for (int num = 0; num < numElements; num += 2)
		{
		// Even - 2n eg. {0,2,4,6, etc.}
		
			// Real
			RFFT[i + num] = (RX[i + num] * (cosf(2 * PI*(i + num) / size))) + (IX[i + num] * 1 * (sinf(2 * PI * (i + num) / size)));
			// Imaginary
			IFFT[i + num] = (RX[i + num] * -1 * (sinf(2 * PI*(i + num) / size))) + (IX[i + num] * (cosf(2 * PI * (i + num) / size)));


		// Odd - 2n-1 eg. {1,3,5,7, etc.}

				// Real
			RFFT[i + num + 1] = (RX[i + num + 1] * (cosf(2 * PI*(i + num + 1) / size))) + (IX[i + num + 1] * 1 * (sinf(2 * PI * (i + num + 1) / size)));
			// Imaginary
			IFFT[i + num + 1] = (RX[i + num + 1] * -1 * (sinf(2 * PI*(i + num + 1) / size))) + (IX[i + num + 1] * (cosf(2 * PI * (i + num + 1) / size)));
		}
	}
}

const int ARRAYSIZE = 32768;

int main()
{

	// Time Check
	clock_t t;
	t = clock();
	srand(time(NULL));

	// FFT declarations
	float RX[ARRAYSIZE];	// Real X values
	float IX[ARRAYSIZE];	// Imaginary X values
	float RFFT[ARRAYSIZE];	// Real FFT components
	float IFFT[ARRAYSIZE];	// Imaginary FFT components
	float RSUM = 0;	// Sum of real components
	float ISUM = 0; // Sum of imaginary components


	// Filling input arrays with numbers from 0.0-10.0
	for (int i = 0; i < ARRAYSIZE; i++)
	{
		RX[i] = (rand() % 100) / 10.0;
		IX[i] = (rand() % 100) / 10.0;
	}

	// FFT Process

	cudaError_t cudaStatus = CudaFFT(RX, IX, RFFT, IFFT, ARRAYSIZE);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CudaFFT failed!");
		return 1;
	}

	for (int i = 0; i < 9999; i++)
	{
		CudaFFT(RX, IX, RFFT, IFFT, ARRAYSIZE);
	}

	// FFT Result

	for (int i = 0; i < ARRAYSIZE; i++)
	{
		RSUM += RFFT[i];
		ISUM += IFFT[i];
	}

	printf("The sum of real components of X is: %f \n The sum of imaginary components of X is: %f \n", RSUM, ISUM);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	// Shows time elapsed
	t = clock() - t;
	double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
	printf("FFT of 32768-element arrays 9999 times took %f seconds to execute using CUDA\n", time_taken);

	return 0;
}


// Helper function for using CUDA to do FFT in parallel 
cudaError_t CudaFFT(const float *RX, const float *IX, float *RFFT, float *IFFT, unsigned int size)
{
	float *dev_RX = 0;
	float *dev_IX = 0;
	float *dev_RFFT = 0;
	float *dev_IFFT = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for four vectors (two input, two output)
	cudaStatus = cudaMalloc((void**)&dev_RX, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_IX, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_RFFT, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_IFFT, size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_RX, RX, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_IX, IX, size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	FFT<<< size/128, 128/4 >>>(dev_RX, dev_IX, dev_RFFT, dev_IFFT, ARRAYSIZE);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "FFT launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(RFFT, dev_RFFT, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(IFFT, dev_IFFT, size * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_RX);
	cudaFree(dev_IX);
	cudaFree(dev_RFFT);
	cudaFree(dev_IFFT);

	return cudaStatus;
}
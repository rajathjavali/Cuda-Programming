// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];
#include<stdio.h>
#include "cuda.h"
#include<string.h>
#include<stdlib.h>

#define BLOCK_SIZE 1024 //@@ You can change this

char *inputFile,*outputFile;
void _errorCheck(cudaError_t e){
	if(e != cudaSuccess){
		printf("Failed to run statement \n");
	}
}

__global__ void totalSequential(float *input, float *output, int len) {
  //@@ Compute reduction for a segment of the input vector
  int tid = threadIdx.x, i = blockIdx.x * blockDim.x;

  if(tid == 0) {
    int sum = 0;
    for(unsigned int j = 0; j <blockDim.x; j++)
    {
      sum += input[i + j];
    }
    output[blockIdx.x] = sum;
  }
}

__global__ void totalSequentialSharedMem(float *input, float *output, int len) {
  //@@ Compute reduction for a segment of the input vector
  int tid = threadIdx.x, i = blockIdx.x * blockDim.x;
  __shared__ float sdata[BLOCK_SIZE];
  sdata[tid] = i + tid < len ? input[i+tid] : 0.0;

  if(tid == 0) {
    for(unsigned int j = 1; j <blockDim.x; j++)
    {
      sdata[0] += sdata[j];
    }
    output[blockIdx.x] = sdata[0];
  }
}

__global__ void totalWithThreadSyncInterleaved(float *input, float *output, int len) {
  //@@ Compute reduction for a segment of the input vector
  int tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;

  for(unsigned int j = 1; j <blockDim.x; j *= 2)
  {
    if (tid % (2 * j) == 0)
      input[i] += input[i+j];
    __syncthreads();
  }

  if(tid == 0) 
  {
    output[blockIdx.x] = input[i];
  }
}

__global__ void totalWithThreadSyncAndSharedMemInterleaved(float *input, float *output, int len) {
  //@@ Compute reduction for a segment of the input vector
  __shared__ float sdata[BLOCK_SIZE];
  int tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(i  < len)
    sdata[tid] = input[i];
  else
    sdata[tid] = 0.0;

  for(unsigned int j = 1; j < blockDim.x; j *= 2)
  {
    if (tid % (2 * j) == 0)
      sdata[tid] += sdata[tid+j];
    __syncthreads();
  }

  if(tid == 0) 
  {
    output[blockIdx.x] = sdata[0];
  }
}

__global__ void totalWithThreadSync(float *input, float *output, int len) {
  //@@ Compute reduction for a segment of the input vector
  int tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;

  for(unsigned int j = blockDim.x/2; j > 0; j = j/2)
  {
    if(tid < j)
    {
      if ((i + j) < len)
      input[i] += input[i+j];
      else
      input [i] += 0.0;
    }
    __syncthreads();
  }

  if(tid == 0) 
  {
    output[blockIdx.x] = input[i];
  }
}

__global__ void totalWithThreadSyncAndSharedMem(float *input, float *output, int len) {
  //@@ Compute reduction for a segment of the input vector
  __shared__ float sdata[BLOCK_SIZE];
  int tid = threadIdx.x, i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(tid < len)
    sdata[tid] = input[i];
  else
    sdata[tid] = 0.0;
  
  __syncthreads();
  
  for(unsigned int j = blockDim.x/2; j > 0; j = j/2)
  {
    if(tid < j)
    {
      sdata[tid] += sdata[tid+j];
    }
    __syncthreads();
  }

  if(tid == 0) 
  {
    output[blockIdx.x] = sdata[0];
  }
}

void parseInput(int argc, char **argv){
	if(argc < 2){
		printf("Not enough arguments\n");
		printf("Usage: reduction -i inputFile -o outputFile\n");	
		exit(1);
	}
	int i=1;
	while(i<argc){
		if(!strcmp(argv[i],"-i")){
			++i;
			inputFile = argv[i];
		}
		else if(!strcmp(argv[i],"-o")){
			++i;
			outputFile = argv[i];
		}
		else{
			printf("Wrong input");
			exit(1);
		}
		i++;
	}
}
void getSize(int &size, char *file){
	FILE *fp;
	fp = fopen(file,"r");
	if(fp == NULL){
		perror("Error opening File\n");
		exit(1);
	}
	
	if(fscanf(fp,"%d",&size)==EOF){
		printf("Error reading file\n");
		exit(1);
	}
	fclose(fp);	
}
void readFromFile(int &size,float *v, char *file){
	FILE *fp;
	fp = fopen(file,"r");
	if(fp == NULL){
		printf("Error opening File %s\n",file);
		exit(1);
	}
	
	if(fscanf(fp,"%d",&size)==EOF){
		printf("Error reading file\n");
		exit(1);
	}
	int i=0;
	float t;
	while(i < size){
		if(fscanf(fp,"%f",&t)==EOF){
			printf("Error reading file\n");
			exit(1);
		}
		v[i++]=t;
		//printf("%lf\t", t);
	}
	fclose(fp);
	

}

int main(int argc, char **argv) {
  int ii;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list
  float *solution;

  // Read arguments and input files
  parseInput(argc,argv);

  // Read input from data
  getSize(numInputElements,inputFile);
  hostInput = (float*) malloc(numInputElements*sizeof(float));
  readFromFile(numInputElements,hostInput,inputFile);  
  
  printf("Data size: %d\tBlock Size: %d\n", numInputElements, BLOCK_SIZE);

  int opsz;
  getSize(opsz,outputFile);	
  solution = (float*) malloc(opsz*sizeof(float));
  readFromFile(opsz,solution,outputFile);
  
  //@@ You can change this, but assumes output element per block
  numOutputElements = numInputElements / (BLOCK_SIZE);
 
  if (numInputElements % (BLOCK_SIZE)) {
    numOutputElements++;
  }

  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  //@@ ------------------------------------------------------------------------------------------------@@//
  //@@ Allocate GPU memory here
  cudaMalloc((void**) &deviceInput, numInputElements * sizeof(float));
  cudaMalloc((void**) &deviceOutput, numOutputElements * sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 grid(numOutputElements, 1, 1);
  dim3 block(BLOCK_SIZE, 1, 1);

  // Initialize timer
  cudaEvent_t start,stop;
  float elapsed_time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  //@@ Launch the GPU Kernel here, you may want multiple implementations to compare
  totalWithThreadSync <<<grid, block>>> (deviceInput, deviceOutput, numInputElements);
  
  cudaDeviceSynchronize();


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

  /*
   * Reduce any remaining output on host
   */
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);        
  cudaEventElapsedTime(&elapsed_time,start, stop);

  if(solution[0] == hostOutput[0]){
  	printf("SUCCESSFUL: with just thread sync:- time = %2.6f\n",elapsed_time);
  }
  else{
	 printf("The operation failed \n");
  }
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  //@@ ------------------------------------------------------------------------------------------------@@//
  //@@ Allocate GPU memory here
  cudaMalloc((void**) &deviceInput, numInputElements * sizeof(float));
  cudaMalloc((void**) &deviceOutput, numOutputElements * sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

  elapsed_time = 0.0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  //@@ Launch the GPU Kernel here, you may want multiple implementations to compare
  totalWithThreadSyncInterleaved<<<grid, block>>> (deviceInput, deviceOutput, numInputElements);
  cudaDeviceSynchronize();


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

  /*
   * Reduce any remaining output on host
   */
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);        
  cudaEventElapsedTime(&elapsed_time,start, stop);

  if(solution[0] == hostOutput[0]){
    printf("SUCCESSFUL: with just thread sync with Interleaved Access:- time = %2.6f\n",elapsed_time);
  }
  else{
   printf("The operation failed \n");
  }
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  //@@ ------------------------------------------------------------------------------------------------@@//
  //@@ Allocate GPU memory here
  cudaMalloc((void**) &deviceInput, numInputElements * sizeof(float));
  cudaMalloc((void**) &deviceOutput, numOutputElements * sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

  elapsed_time = 0.0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  //@@ Launch the GPU Kernel here, you may want multiple implementations to compare
  totalWithThreadSyncAndSharedMem <<<grid, block>>> (deviceInput, deviceOutput, numInputElements);
  cudaDeviceSynchronize();


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

  /*
   * Reduce any remaining output on host
   */
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);        
  cudaEventElapsedTime(&elapsed_time,start, stop);

  if(solution[0] == hostOutput[0]){
    printf("SUCCESSFUL: with just thread sync and shared memory:- time = %2.6f\n",elapsed_time);
  }
  else{
   printf("The operation failed \n");
  }
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  //@@ ------------------------------------------------------------------------------------------------@@//
  //@@ Allocate GPU memory here
  cudaMalloc((void**) &deviceInput, numInputElements * sizeof(float));
  cudaMalloc((void**) &deviceOutput, numOutputElements * sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

  elapsed_time = 0.0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  //@@ Launch the GPU Kernel here, you may want multiple implementations to compare
  totalWithThreadSyncAndSharedMemInterleaved<<<grid, block>>> (deviceInput, deviceOutput, numInputElements);
  cudaDeviceSynchronize();


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

  /*
   * Reduce any remaining output on host
   */
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);        
  cudaEventElapsedTime(&elapsed_time,start, stop);

  if(solution[0] == hostOutput[0]){
    printf("SUCCESSFUL: with just thread sync and shared memory with Interleaved Access:- time = %2.6f\n",elapsed_time);
  }
  else{
   printf("The operation failed \n");
  }
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  //@@ ------------------------------------------------------------------------------------------------@@//
  //@@ Allocate GPU memory here
  cudaMalloc((void**) &deviceInput, numInputElements * sizeof(float));
  cudaMalloc((void**) &deviceOutput, numOutputElements * sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

  elapsed_time = 0.0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  //@@ Launch the GPU Kernel here, you may want multiple implementations to compare
  totalSequential <<<grid, block>>> (deviceInput, deviceOutput, numInputElements);
  cudaDeviceSynchronize();


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

  /*
   * Reduce any remaining output on host
   */
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);        
  cudaEventElapsedTime(&elapsed_time,start, stop);

  if(solution[0] == hostOutput[0]){
    printf("SUCCESSFUL: with just thread 0 of every block computing:- time = %2.6f\n",elapsed_time);
  }
  else{
   printf("The operation failed \n");
  }
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  //@@ ------------------------------------------------------------------------------------------------@@//
  //@@ Allocate GPU memory here
  cudaMalloc((void**) &deviceInput, numInputElements * sizeof(float));
  cudaMalloc((void**) &deviceOutput, numOutputElements * sizeof(float));

  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

  elapsed_time = 0.0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  //@@ Launch the GPU Kernel here, you may want multiple implementations to compare
  totalSequentialSharedMem <<<grid, block>>> (deviceInput, deviceOutput, numInputElements);
  cudaDeviceSynchronize();


  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

  /*
   * Reduce any remaining output on host
   */
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);        
  cudaEventElapsedTime(&elapsed_time,start, stop);

  if(solution[0] == hostOutput[0]){
    printf("SUCCESSFUL: with just thread 0 of every block computing using shared memory:- time = %2.6f\n",elapsed_time);
  }
  else{
   printf("The operation failed \n");
  }
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  printf("____________________________________________________________________\n\n\n");

  free(hostInput);
  free(hostOutput);

  return 0;
}

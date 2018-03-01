#include<stdio.h>
#include "cuda.h"
#include<string.h>
#include<stdlib.h>
#include <fcntl.h>
#include <unistd.h>

#define DEFAULT_THRESHOLD  4000

#define _TILESIZE_ 30
#define _TILESIZE_2 32

//#define DEFAULT_FILENAME "BWstop-sign.ppm"
#define DEFAULT_FILENAME "mountains.ppm"

int blocksize = 0;
void (*func)(int *, int *, int , int , int);

unsigned int *read_ppm( char *filename, int & xsize, int & ysize, int & maxval ){
  
  if ( !filename || filename[0] == '\0') {
    fprintf(stderr, "read_ppm but no file name\n");
    return NULL;  // fail
  }

  fprintf(stderr, "read_ppm( %s )\n", filename);
  int fd = open( filename, O_RDONLY);
  if (fd == -1) 
    {
      fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
      return NULL; // fail 

    }

  char chars[1024];
  int num = read(fd, chars, 1000);

  if (chars[0] != 'P' || chars[1] != '6') 
    {
      fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
      return NULL;
    }

  unsigned int width, height, maxvalue;


  char *ptr = chars+3; // P 6 newline
  if (*ptr == '#') // comment line! 
    {
      ptr = 1 + strstr(ptr, "\n");
    }

  num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
  fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
  xsize = width;
  ysize = height;
  maxval = maxvalue;
  
  unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
  if (!pic) {
    fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
    return NULL; // fail but return
  }

  // allocate buffer to read the rest of the file into
  int bufsize =  3 * width * height * sizeof(unsigned char);
  if (maxval > 255) bufsize *= 2;
  unsigned char *buf = (unsigned char *)malloc( bufsize );
  if (!buf) {
    fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
    return NULL; // fail but return
  }

  // TODO really read
  char duh[80];
  char *line = chars;

  // find the start of the pixel data.   no doubt stupid
  sprintf(duh, "%d\0", xsize);
  line = strstr(line, duh);
  //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
  line += strlen(duh) + 1;

  sprintf(duh, "%d\0", ysize);
  line = strstr(line, duh);
  //fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
  line += strlen(duh) + 1;

  sprintf(duh, "%d\0", maxval);
  line = strstr(line, duh);


  fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
  line += strlen(duh) + 1;

  long offset = line - chars;
  lseek(fd, offset, SEEK_SET); // move to the correct offset
  long numread = read(fd, buf, bufsize);
  fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize); 

  close(fd);


  int pixels = xsize * ysize;
  for (int i=0; i<pixels; i++) pic[i] = (int) buf[3*i];  // red channel

 

  return pic; // success
}

void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic) 
{
 FILE *fp;
 
 fp = fopen(filename, "w");
 if (!fp) 
   {
     fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n");
     exit(-1); 
   }
 
 
 fprintf(fp, "P6\n"); 
 fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);
 
 int numpix = xsize * ysize;
 for (int i=0; i<numpix; i++) {
   unsigned char uc = (unsigned char) pic[i];
   fprintf(fp, "%c%c%c", uc, uc, uc); 
 }
 fclose(fp);

}

__global__ void sobelEdgeDetection(int *input, int *output, int width, int height, int thresh) {
  
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = j * width + i;

  if ( ((i > 0) && (j > 0)) && ((i < (width - 1)) && (j < (height - 1))))
  {
    
    int sum1 = 0, sum2 = 0, magnitude;

    sum1 = input[width * (j - 1) + (i + 1)] -     input[width * (j - 1) + (i - 1)]
     + 2 * input[width * (j)     + (i + 1)] - 2 * input[width * (j)     + (i - 1)]
     +     input[width * (j + 1) + (i + 1)] -     input[width * (j + 1) + (i - 1)];

    sum2 = input[width * (j - 1) + (i - 1)] + 2 * input[width * (j - 1) + (i)] + input[width * (j - 1) + (i + 1)]
         - input[width * (j + 1) + (i - 1)] - 2 * input[width * (j + 1) + (i)] - input[width * (j + 1) + (i + 1)];

    magnitude = sum1 * sum1 + sum2 * sum2;
    if(magnitude > thresh)
      output[index] = 255;
    else
      output[index] = 0;
  }
}

__global__ void sobelEdgeDetectionWithRegisters (int *input, int *output, int width, int height, int thresh) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = j * width + i;

  int val1 = input[width * (j - 1) + (i + 1)], val2 = input[width * (j - 1) + (i - 1)], val3 = input[width * (j + 1) + (i + 1)], val4 = input[width * (j + 1) + (i - 1)];

  if ( ((i > 0) && (j > 0)) && ((i < (width - 1)) && (j < (height - 1))))
  {
   
    int sum1 = 0, sum2 = 0, magnitude;

    sum1 = val1 - val2
    + 2 * input[width * (j)     + (i + 1)] - 2 * input[width * (j)     + (i - 1)]
    +     val3 - val4;

    sum2 = val2 + 2 * input[width * (j - 1) + (i)] + val1
        - val4 - 2 * input[width * (j + 1) + (i)] - val3;

    magnitude = sum1 * sum1 + sum2 * sum2;
    if(magnitude > thresh)
      output[index] = 255;
    else
      output[index] = 0;
  }
  else {
    output[index] = 0;  
  }
}

__global__ void sobelEdgeDetectionSharedMem(int *input, int *output, int width, int height, int thresh) {
 
  int blockSize = 32;
  static __shared__ int shMem[34][34];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int index = j * width + i;

  int xind = threadIdx.x + 1;
  int yind = threadIdx.y + 1;

  shMem[xind][yind] = input[width * j + i];

  if ( i > 0 && j > 0 && i < width - 1 && j < height - 1)
  {
    if(threadIdx.x == 0)
    shMem[xind-1][yind] = input[width * j + i-1];

    if(threadIdx.y == 0)
    shMem[xind][yind-1] = input[width * (j-1) + i];

    if(threadIdx.x == blockSize+1)
    shMem[xind+1][yind] = input[width * j + i+1];

    if(threadIdx.y == blockSize+1)
    shMem[xind][yind+1] = input[width * (j+1) + i];

    if(threadIdx.x == 0 && threadIdx.y == 0)
    shMem[xind-1][yind-1] = input[width * (j-1) + i-1];

    if(threadIdx.x == blockSize+1 && threadIdx.y == 0)
    shMem[xind+1][yind-1] = input[width * (j-1) + i+1];

    if(threadIdx.x == 0 && threadIdx.y == blockSize+1)
    shMem[xind-1][yind+1] = input[width * (j+1) + i-1];

    if(threadIdx.x == blockSize+1 && threadIdx.y == blockSize+1)
    shMem[xind+1][yind+1] = input[width * (j+1) + i+1];
  }
  __syncthreads();


    int sum1 = 0, sum2 = 0, magnitude;

    sum1 = shMem[xind+1][yind-1] -     shMem[xind-1][yind-1]
    + 2 * shMem[xind+1][yind  ] - 2 * shMem[xind-1][yind  ]
    +     shMem[xind+1][yind+1] -     shMem[xind-1][yind+1];

    sum2 = shMem[xind-1][yind-1] + 2 * shMem[xind][yind-1] + shMem[xind+1][yind-1]
        - shMem[xind-1][yind+1] - 2 * shMem[xind][yind+1] - shMem[xind+1][yind+1];

    magnitude = sum1 * sum1 + sum2 * sum2;
    if(magnitude > thresh)
      output[index] = 255;
    else
      output[index] = 0;

}

__global__ void sobelEdgeDetectionSharedMem2(int *input, int *output, int width, int height, int thresh) {

  int regArr[4][4];

  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
  int j = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

  if ( i > 0 && j > 0 && i < width - 1 && j < height - 1)
  {

    regArr[0][0] = input[width * (j-1) + i - 1];
    regArr[0][1] = input[width * (j-1) + i    ];
    regArr[0][2] = input[width * (j-1) + i + 1];
    regArr[0][3] = input[width * (j-1) + i + 2];
    regArr[1][0] = input[width * (j)   + i - 1];
    regArr[1][1] = input[width * (j)   + i    ];
    regArr[1][2] = input[width * (j)   + i + 1];
    regArr[1][3] = input[width * (j)   + i + 2];
    regArr[2][0] = input[width * (j+1) + i - 1];
    regArr[2][1] = input[width * (j+1) + i    ];
    regArr[2][2] = input[width * (j+1) + i + 1];
    regArr[2][3] = input[width * (j+1) + i + 2];
    regArr[3][0] = input[width * (j+2) + i - 1];
    regArr[3][1] = input[width * (j+2) + i    ];
    regArr[3][2] = input[width * (j+2) + i + 1];
    regArr[3][3] = input[width * (j+2) + i + 2];
   
    __syncthreads();

    
    int sum1 = 0, sum2 = 0, magnitude;
    int num = 3;

    for(int xind = 1; xind < num; xind++)
    {
      for(int yind = 1; yind < num; yind++)
      {
        sum1 = regArr[xind+1][yind-1] -     regArr[xind-1][yind-1]
         + 2 * regArr[xind+1][yind  ] - 2 * regArr[xind-1][yind  ]
         +     regArr[xind+1][yind+1] -     regArr[xind-1][yind+1];

        sum2 = regArr[xind-1][yind-1] + 2 * regArr[xind][yind-1] + regArr[xind+1][yind-1]
             - regArr[xind-1][yind+1] - 2 * regArr[xind][yind+1] - regArr[xind+1][yind+1];

        magnitude = sum1 * sum1 + sum2 * sum2;
        
        if(magnitude > thresh)
          output[(j + yind - 1) * width + (i + xind - 1)] = 255;
        else
          output[(j + yind - 1) * width + (i + xind - 1)] = 0;

      }
    } 
  }
}

__global__ void sobelEdgeDetectionSharedMemUnrollControlFlow(int *input, int *output, int width, int height, int thresh) {
 
  unsigned int blockSize = 32;
  static __shared__ int shMem[34][34];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  int xind = threadIdx.x + 1;
  int yind = threadIdx.y + 1;

  shMem[xind][yind] = input[width * j + i];

  if ( i > 0 && j > 0 && i < width - 1 && j < height - 1)
  {
    if(threadIdx.x == 0)
    shMem[xind-1][yind] = input[width * j + i-1];

    if(threadIdx.y == 0)
    shMem[xind][yind-1] = input[width * (j-1) + i];

    if(threadIdx.x == blockSize+1)
    shMem[xind+1][yind] = input[width * j + i+1];

    if(threadIdx.y == blockSize+1)
    shMem[xind][yind+1] = input[width * (j+1) + i];

    if(threadIdx.x == 0 && threadIdx.y == 0)
    shMem[xind-1][yind-1] = input[width * (j-1) + i-1];

    if(threadIdx.x == blockSize+1 && threadIdx.y == 0)
    shMem[xind+1][yind-1] = input[width * (j-1) + i+1];

    if(threadIdx.x == 0 && threadIdx.y == blockSize+1)
    shMem[xind-1][yind+1] = input[width * (j+1) + i-1];

    if(threadIdx.x == blockSize+1 && threadIdx.y == blockSize+1)
    shMem[xind+1][yind+1] = input[width * (j+1) + i+1];
  }
  __syncthreads();


  int sum1 = 0, sum2 = 0, magnitude;
  int num = 3;

  for(int xind = 1; xind < num; xind++)
  {
    for(int yind = 1; yind < num; yind++)
    {
      sum1 = shMem[xind+1][yind-1] -     shMem[xind-1][yind-1]
       + 2 * shMem[xind+1][yind  ] - 2 * shMem[xind-1][yind  ]
       +     shMem[xind+1][yind+1] -     shMem[xind-1][yind+1];

      sum2 = shMem[xind-1][yind-1] + 2 * shMem[xind][yind-1] + shMem[xind+1][yind-1]
           - shMem[xind-1][yind+1] - 2 * shMem[xind][yind+1] - shMem[xind+1][yind+1];

      magnitude = sum1 * sum1 + sum2 * sum2;
      
      if(magnitude > thresh)
        output[(j + yind - 1) * width + (i + xind - 1)] = 255;
      else
        output[(j + yind - 1) * width + (i + xind - 1)] = 0;

    }
  } 

}

__global__ void sobelEdgeDetectionSharedMemUnroll(int *input, int *output, int width, int height, int thresh) {

  __shared__ int shMem[4 * _TILESIZE_2 * _TILESIZE_2 ];

  unsigned int size = 2 * _TILESIZE_2;
  int num = 2;

  int i = blockIdx.x * num * _TILESIZE_ + threadIdx.x * num;
  int j = blockIdx.y * num * _TILESIZE_ + threadIdx.y * num;

  int xind = num * threadIdx.x;
  int yind = num * threadIdx.y;

  for(int x = 0; x < num; x++)
  {
    for(int y = 0; y < num; y++)
    {
      shMem[ size * (yind + y) + (xind + x)] = input[(j + y) * width + (i + x)];
    }
  }

  __syncthreads();

  if ( xind > 0 && yind > 0 && xind < (size - 2) && yind < (size - 2))
  {
    for(int x = 0; x < num; x++)
    {
      for(int y = 0; y < num; y++)
      { 
        
        int sum1 = shMem[(xind + 1 + x) + size * (yind - 1 + y)] -     shMem[(xind - 1 + x) + size * (yind - 1 + y)]
             + 2 * shMem[(xind + 1 + x) + size * (yind     + y)] - 2 * shMem[(xind - 1 + x) + size * (yind     + y)]
             +     shMem[(xind + 1 + x) + size * (yind + 1 + y)] -     shMem[(xind - 1 + x) + size * (yind + 1 + y)];

        int sum2 = shMem[(xind - 1 + x) + size * (yind - 1 + y)] + 2 * shMem[(xind     + x) + size * (yind - 1 + y)] + shMem[(xind + 1 + x) + size * (yind - 1 + y)]
                 - shMem[(xind - 1 + x) + size * (yind + 1 + y)] - 2 * shMem[(xind     + x) + size * (yind + 1 + y)] - shMem[(xind + 1 + x) + size * (yind + 1 + y)];

        int magnitude = sum1 * sum1 + sum2 * sum2;
        
        int index = (j + y) * width + (i + x);

        if(magnitude > thresh)
          output[index] = 255;
        else
          output[index] = 0;
        
      }
    } 
  }

}

__global__ void sobelEdgeDetectionSharedMemOverlap(int *input, int *output, int width, int height, int thresh) {

  static __shared__ int shMem[_TILESIZE_2 * _TILESIZE_2];

  unsigned int blocksize = _TILESIZE_2;
  int i = blockIdx.x * (blocksize - 2) + threadIdx.x;
  int j = blockIdx.y * (blocksize - 2) + threadIdx.y;
  int index = j * width + i;

  int xind = threadIdx.x;
  int yind = threadIdx.y;
  
  shMem[blocksize * yind + xind] = input[index];
  __syncthreads();
  
  if ( xind > 0 && yind > 0 && xind < (blocksize - 1) && yind < (blocksize - 1))
  {
    
    int sum1 = shMem[xind+1 + blocksize * (yind-1)] -     shMem[xind-1 + blocksize * (yind-1)]
         + 2 * shMem[xind+1 + blocksize * (yind  )] - 2 * shMem[xind-1 + blocksize * (yind  )]
         +     shMem[xind+1 + blocksize * (yind+1)] -     shMem[xind-1 + blocksize * (yind+1)];

    int sum2 = shMem[xind-1 + blocksize * (yind-1)] + 2 * shMem[xind + blocksize * (yind-1)] + shMem[xind+1 + blocksize * (yind-1)]
             - shMem[xind-1 + blocksize * (yind+1)] - 2 * shMem[xind + blocksize * (yind+1)] - shMem[xind+1 + blocksize * (yind+1)];

    int magnitude = sum1 * sum1 + sum2 * sum2;
    if(magnitude > thresh)
      output[index] = 255;
    else
      output[index] = 0;
  }
}

int main(int argc, char **argv) {

  int thresh = DEFAULT_THRESHOLD;
  char *filename;
  filename = strdup( DEFAULT_FILENAME);
  
  if (argc > 1) {
    if (argc == 3)  { // filename AND threshold
      filename = strdup( argv[1]);
       thresh = atoi( argv[2] );
    }
    if (argc == 2) { // default file but specified threshhold
      
      thresh = atoi( argv[1] );
    }

    fprintf(stderr, "file %s    threshold %d\n", filename, thresh); 
  }

  int xsize, ysize, maxval;
  unsigned int *pic = read_ppm( filename, xsize, ysize, maxval ); 

  char resultFile[50];
  int size = strlen(filename) - 4;

  strncpy(resultFile, filename, size);
  resultFile[size] = '\0';

  int numbytes =  xsize * ysize * 3 * sizeof( int );
  int *result = (int *) malloc( numbytes );
  if (!result) { 
   fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
   exit(-1); // fail
  }


  //@@ ------------------------------------------------------------------------------------------------@@//

  int *deviceInput, *deviceOutput;
  int imgSize = xsize * ysize * sizeof(unsigned int);
  printf("width %d height %d imgsize %d\n", xsize, ysize, imgSize);
  //@@ Allocating device memory for both input and output images
  cudaMalloc((void**) &deviceInput, imgSize);
  cudaMalloc((void**) &deviceOutput, imgSize);

  //@@ Copying input image from host memory to the GPU here
  cudaMemcpy(deviceInput, pic, imgSize, cudaMemcpyHostToDevice);

 
  //@@ Initialize the grid and block dimensions here
  dim3 num_threads(_TILESIZE_2, _TILESIZE_2, 1);
  dim3 num_threads_norm(_TILESIZE_, _TILESIZE_, 1);
  dim3 num_blocks(ceil(xsize / _TILESIZE_), ceil(ysize / _TILESIZE_), 1);
  dim3 num_blocks_half(ceil(xsize / (2 * _TILESIZE_)), ceil(ysize / (2 * _TILESIZE_)), 1);


  // Initialize timer
  cudaEvent_t start,stop;
  float elapsed_time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  printf ("\n\n~~~~~~ Performace Testing (Shared Unroll) on %s ~~~~~~~~~~\n\n", filename);
  //printf ("\n\n~~~~~~ Performace Testing (Shared Overlap) on %s ~~~~~~~~~~\n\n", filename);
  //printf ("\n\n~~~~~~ Performace Testing (Global Memory) on %s ~~~~~~~~~~\n\n", filename);
  //printf ("\n\n~~~~~~ Performace Testing (Shared Memory) on %s ~~~~~~~~~~\n\n", filename);
  //printf ("\n\n~~~~~~ Performace Testing (Unroll) on %s ~~~~~~~~~~\n\n", filename);
  //int i = 0;
  //for(i; i < 4; i++)
  {
    char finalFile[50];
    //char ch[2];
    //ch[0] = i + '0';
    //ch[1] = '\0';
    strcpy(finalFile, resultFile);
    strcat(finalFile, "SharedUnroll");
    //strcat(finalFile, "SharedOverlap");
    //strcat(finalFile, "Unroll");
    //strcat(finalFile, "Global");
    //strcat(finalFile, "Shared");
    strcat(finalFile, ".ppm");

    cudaEventRecord(start,0);
   
    //sobelEdgeDetection <<<num_blocks, num_threads_norm>>> (deviceInput, deviceOutput, xsize, ysize, thresh);
    //sobelEdgeDetectionSharedMem <<<num_blocks, num_threads_norm>>> (deviceInput, deviceOutput, xsize, ysize, thresh);
    //sobelEdgeDetectionWithRegisters <<<num_blocks, num_threads>>> (deviceInput, deviceOutput, xsize, ysize, thresh);      
    //sobelEdgeDetectionSharedMemOverlap <<<num_blocks, num_threads>>> (deviceInput, deviceOutput, xsize, ysize, thresh);
    //sobelEdgeDetectionSharedMem2 <<<num_blocks_half, num_threads>>> (deviceInput, deviceOutput, xsize, ysize, thresh);
    sobelEdgeDetectionSharedMemUnroll <<<num_blocks_half, num_threads>>> (deviceInput, deviceOutput, xsize, ysize, thresh);
    /*switch(i){

    case 0:
      printf("~~~~~~~~~~~~~~~ Sobel Edge Detection Basic ~~~~~~~~~~~~\n");
      func = &sobelEdgeDetection;
      sobelEdgeDetection <<<num_blocks, num_threads>>> (deviceInput, deviceOutput, xsize, ysize, thresh);
      break;
    case 1:
      printf("~~~~~~~~~~~~~~~ Sobel Edge Detection With Shared Mem ~~~~~~~~\n");
      func = &sobelEdgeDetectionSharedMem;
      sobelEdgeDetectionSharedMem <<<num_blocks, num_threads>>> (deviceInput, deviceOutput, xsize, ysize, thresh);
      break;
    case 2:
      printf("~~~~~~~~~~~~~~~ Sobel Edge Detection With Registers ~~~~~~~~\n");
      func = &sobelEdgeDetectionWithRegisters;
      sobelEdgeDetectionWithRegisters <<<num_blocks, num_threads>>> (deviceInput, deviceOutput, xsize, ysize, thresh);
      break;
    case 3:
      printf("~~~~~~~~~~~~~~~ Sobel Edge Detection unroll ~~~~~~~~\n");
      func = &sobelEdgeDetectionSharedMem2;
      sobelEdgeDetectionSharedMem2 <<<num_blocks, num_threads>>> (deviceInput, deviceOutput, xsize, ysize, thresh);
      break;
    }*/
    //@@ Launch the GPU Kernel here, you may want multiple implementations to compare
    //func <<<num_blocks, num_threads>>> (deviceInput, deviceOutput, xsize, ysize, thresh);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);        
    cudaEventElapsedTime(&elapsed_time,start, stop);

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(result, deviceOutput, imgSize, cudaMemcpyDeviceToHost);

    printf("Elapsed time: %f\n", elapsed_time);
    write_ppm( finalFile, xsize, ysize, 255, result);
  
  }

  printf("_________________________________________________________________________________\n\n\n");
  
  //@@ Free the GPU memory here
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  free(result);
  free(pic);

  return 0;
}


#include "cuda_runtime.h"
#include <stdint.h>

__global__ void row_contiguous_kernel(float* dst, const uint8_t* src,
                                      int width, int height)
{
  int x = blockDim.x*blockIdx.x + threadIdx.x;
  int y = blockDim.y*blockIdx.y + threadIdx.y;
  int c = blockIdx.z;
  int channels = gridDim.z;

  if(x >= width || y >= height)
    return;

  dst[(c*height + y)*width + x] = 1.f/255.f * src[(y*width + x)*channels + c];
}

void make_row_contiguous(float* dst, const uint8_t* src,
                         int width, int height, int channels)
{
  int block = 32;
  dim3 gridDim((width+block-1)/block, (height+block-1)/block, channels);
  dim3 blockDim(block, block, 1);
  row_contiguous_kernel<<<gridDim, blockDim>>>(dst, src, width, height);
}

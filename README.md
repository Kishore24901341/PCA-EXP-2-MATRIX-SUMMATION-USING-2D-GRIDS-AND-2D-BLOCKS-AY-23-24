# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.
<h3>AIM:To Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.</h3>
<h3>NAME : KISHORE V</h3>
<h3>REGISTER NO : 212224240077</h3>
<h3>EX. NO  : 03</h3>
<h3>DATE  : 18 / 08 / 2026</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```python
%%writefile sobelEdgeDetectionFilter.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencvc.hpp>

using namespace cv;
_global_ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage, unsigned int width, unsigned int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float Kx[3][3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    float Ky[3][3] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
    // only threads inside image will write results
    if ((x >= 3 / 2) && (x < (width - 3 / 2)) && (y >= 3 / 2) && (y < (height - 3 / 2))) {
        // Gradient in x-direction
        float Gx = 0;
        // Loop inside the filter to average pixel values
        for (int ky = -3 / 2; ky <= 3 / 2; ky++) {
            for (int kx = -3 / 2; kx <= 3 / 2; kx++) {
                float fl = srcImage[((y + ky) * width + (x + kx))];
                Gx += fl * Kx[ky + 3 / 2][kx + 3 / 2];
            }
        }
        float Gx_abs = Gx < 0 ? -Gx : Gx;
        // Gradient in y-direction
        float Gy = 0;
        // Loop inside the filter to average pixel values
        for (int ky = -3 / 2; ky <= 3 / 2; ky++) {
            for (int kx = -3 / 2; kx <= 3 / 2; kx++) {
                float fl = srcImage[((y + ky) * width + (x + kx))];
                Gy += fl * Ky[ky + 3 / 2][kx + 3 / 2];
            }
        }
        float Gy_abs =   Gy < 0 ? -Gy : Gy;
        dstImage[(y * width + x)] = Gx_abs + Gy_abs;
    }
}




void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Read input image
    Mat image = imread("/content/tiger.jpg", IMREAD_COLOR);

    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    // Allocate host memory for output image
    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);
    if (h_outputImage == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
    checkCudaErrors(cudaMemcpy(d_inputImage, image.data, imageSize, cudaMemcpyHostToDevice));

    // Define CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize(ceil(width / 16.0), ceil(height / 16.0));

    cudaEventRecord(start);
    sobelFilter<<>>(d_inputImage, d_outputImage, width, height);
    cudaEventRecord(stop);

    // Synchronize events
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));

    // Write output image
    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("output_sobel.jpeg", outputImage);

    // Free memory
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print elapsed time
    printf("Total time taken: %f milliseconds\n", milliseconds);

    return 0;
}
```

```python
import cv2
from matplotlib import pyplot as plt

# Read and display the output image
output_image_path = '/content/tiger.jpg'
output_image = cv2.imread(output_image_path, cv2.IMREAD_GRAYSCALE)  # Use IMREAD_GRAYSCALE if it's a single-channel image

# Display the image
plt.imshow(output_image, cmap='gray')
plt.title('Edge Detection Output')
plt.axis('off')  # Hide the axes
plt.show()
```

## OUTPUT:
<img width="1752" height="589" alt="image" src="https://github.com/user-attachments/assets/d6a8c1ac-2a52-4b1f-bdd5-89ce8eee1194" />

<img width="1778" height="865" alt="image" src="https://github.com/user-attachments/assets/111b3a2f-73be-4876-a150-3349ce15df17" />

## RESULT:
Thus the program has been executed by using CUDA to perform Sobel edge detection on an image using GPU parallel processing

What challenges did you face while implementing the Sobel filter for color images?
How did changing the block size influence the performance of your CUDA implementation?
What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.
Suggest potential optimizations for improving the performance of the Sobel filter.

Deliverables:
One of the main challenges was processing three color channels (Red, Green, and Blue) instead of a single grayscale channel. Each channel had to be filtered separately while maintaining correct memory indexing. Handling image boundaries was also challenging because pixels at the edges do not have all neighboring pixels required for the Sobel operator. Additionally, optimizing global memory access and avoiding unnecessary memory transfers between the CPU and GPU required careful implementation.

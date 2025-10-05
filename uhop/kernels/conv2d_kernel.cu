// uhop/kernels/conv2d_kernel.cu   

__global__ void conv2d_kernel(
    const float* input, const float* kernel, 
    float* output,
    int in_channels, int in_height, int in_width,
    int out_channels, int kernel_size,
    int out_height, int out_width,
    int stride, int padding
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c_out = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < out_width && y < out_height && c_out < out_channels) {
        float sum = 0.0f;
        
        for (int c_in = 0; c_in < in_channels; c_in++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_y = y * stride + ky - padding;
                    int in_x = x * stride + kx - padding;
                    
                    if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                        int input_idx = c_in * (in_height * in_width) + in_y * in_width + in_x;
                        int kernel_idx = c_out * (in_channels * kernel_size * kernel_size) + 
                                        c_in * (kernel_size * kernel_size) + 
                                        ky * kernel_size + kx;
                        
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        
        int output_idx = c_out * (out_height * out_width) + y * out_width + x;
        output[output_idx] = sum;
    }
}
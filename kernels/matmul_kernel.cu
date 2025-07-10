// kernels/matmul_kernel.cu

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
    int M, int N, int K) {
// Calculate global thread coordinates
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if (row < M && col < N) {
float sum = 0.0f;
for (int i = 0; i < K; ++i) {
sum += A[row * K + i] * B[i * N + col];
}
C[row * N + col] = sum;
}
}
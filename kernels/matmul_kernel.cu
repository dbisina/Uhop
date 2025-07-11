// kernels/matmul_kernel.cu

__global__ void matmul_kernel(const float* A, const float* B, float* C, 
    int M, int N, int K) {
__shared__ float sA[16][16];
__shared__ float sB[16][16];

int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
float sum = 0.0f;

for (int tile = 0; tile < (K + blockDim.x - 1)/blockDim.x; ++tile) {
    int tX = tile * blockDim.x + threadIdx.x;
    int tY = tile * blockDim.y + threadIdx.y;

    sA[threadIdx.y][threadIdx.x] = (row < M && tX < K) 
                    ? A[row*K + tX] : 0;
    sB[threadIdx.x][threadIdx.y] = (tY < K && col < N)
                    ? B[tY*N + col] : 0;
    __syncthreads();

    for (int i = 0; i < blockDim.x; ++i) {
        sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
    }
    __syncthreads();
    }

if (row < M && col < N) {
C[row*N + col] = sum;
}
}
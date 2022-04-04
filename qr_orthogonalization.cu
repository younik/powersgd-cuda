#include <cub/cub.cuh>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ void wait_barrier(int* barrier, int target){
    if (threadIdx.x == 0){
        int counter;
        do {
            asm volatile ("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(counter): "l"(barrier) );
        }
        while (counter < target);
    }
    __syncthreads();
}

__device__ __forceinline__ void set_barrier(int* barrier, int value){
    if(threadIdx.x == 0)
        asm volatile ("st.global.cg.s32 [%0], %1;" :: "l"(barrier), "r"(value));
}

template <int BLOCK_THREADS, typename scalar_t>
__device__  __forceinline__ scalar_t dot(scalar_t *a, scalar_t *b, uint length){
    typedef cub::BlockReduce<scalar_t, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    int tx = threadIdx.x;
    uint unroll = ceil( (float)length / (float)BLOCK_THREADS );
    uint idx = (tx & -32u)*unroll + (tx & 31);

    scalar_t localProd = 0;
    for (uint i = 0; i < unroll; ++i){
        localProd += (idx < length)? a[idx] * b[idx] : (scalar_t) 0;
        idx += 32;
    }

    scalar_t reduce = BlockReduce(tmpStorage).Sum(localProd);

     __shared__ scalar_t dot;
    if (tx == 0) 
        dot = reduce;
    __syncthreads();

    return dot;
}

template <int BLOCK_THREADS, typename scalar_t> 
__global__ void reflections(scalar_t *R, scalar_t *vs, int m, int n, int *barriers){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    scalar_t *v = &vs[bx * n + bx];
    uint vLen = n - bx;

    wait_barrier(&barriers[bx], bx);

    for (uint idx = tx; idx < vLen; idx += BLOCK_THREADS)
        v[idx] = - R[bx * n + bx + idx];

    scalar_t normVSq = dot<BLOCK_THREADS, scalar_t>(v, v, vLen);
    if (tx == 0) 
        v[0] += copysign(sqrt(normVSq), v[0]);
    
    scalar_t normV = sqrt(dot<BLOCK_THREADS, scalar_t>(v, v, vLen));
    for (uint idx = tx; idx < vLen; idx += BLOCK_THREADS)
        v[idx] /= normV;

    for (uint row = bx + 1; row < m; ++row){
        wait_barrier(&barriers[row], bx);

        scalar_t dotValue = dot<BLOCK_THREADS, scalar_t>(&R[row * n + bx], v, vLen);
        
        for (uint idx = tx; idx < vLen; idx += BLOCK_THREADS)
            R[row * n + bx + idx] -= 2.0 * v[idx] * dotValue;

        __syncthreads();
        set_barrier(&barriers[row], bx + 1);
    }
}

template <int BLOCK_THREADS, typename scalar_t> 
__global__  void QLoop(scalar_t *Q, scalar_t *vs, int n, int m){
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    for (int vIdx = 0; vIdx < m; ++vIdx){
        scalar_t *v = &vs[vIdx * n + vIdx];
        uint vLen = n - vIdx;
    
        scalar_t dotValue = dot<BLOCK_THREADS, scalar_t>(v, &Q[bx * n + vIdx], vLen);

        for (uint idx = tx; idx < vLen ; idx += BLOCK_THREADS)
            Q[bx * n + vIdx + idx] -= 2.0 * v[idx] * dotValue;

        __syncthreads();
    }
}

template <int BLOCK_THREADS, typename scalar_t> 
void qrMain(torch::Tensor A, torch::Tensor out, int m, int n, float epsilon){
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto options = torch::TensorOptions().dtype(torch::kInt32).device(A.device());
    torch::Tensor barriers = torch::zeros({m}, options);
    
    torch::Tensor vs = torch::zeros_like(A);
    torch::Tensor R = A.clone();
    R.diagonal().add_((scalar_t) epsilon);
    
    reflections<BLOCK_THREADS, scalar_t><<<m, BLOCK_THREADS, 0, stream>>>(R.data<scalar_t>(), vs.data<scalar_t>(), m, n, barriers.data<int>());

    out.fill_(0);
    out.fill_diagonal_(1);
    QLoop<BLOCK_THREADS, scalar_t><<<m, BLOCK_THREADS, 0, stream>>>(out.data<scalar_t>(), vs.data<scalar_t>(), n, m);
}

template <typename scalar_t> 
void typedImplementation(torch::Tensor A, torch::Tensor out, int m, int n, float epsilon){
    if (n < 512)
        return qrMain<256, scalar_t>(A, out, m, n, epsilon);
    else if (n < 1024)
        return qrMain<512, scalar_t>(A, out, m, n, epsilon);
    else
        return qrMain<1024, scalar_t>(A, out, m, n, epsilon);
}

void qrOrthogonalizationCuda(torch::Tensor A, torch::Tensor out, int m, int n, float epsilon){
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
    A.scalar_type(), "qr_orthogonalization_cuda", ([&] {
        typedImplementation<scalar_t>(A, out, m, n, epsilon);
    }));
}
#include <cub/cub.cuh>
#include <cuda/std/semaphore>
#include <torch/extension.h>

using semaphore = cuda::std::counting_semaphore<>;
const int BLOCK_THREADS = 512;


template <int BLOCK_THREADS, typename scalar_t>
__device__  scalar_t dot(scalar_t *a, scalar_t *b, int length, int tx){
    typedef cub::BlockReduce<scalar_t, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    int loopTimes = ceil((float)length / (float)BLOCK_THREADS);
    __shared__ scalar_t dot;
    if(tx == 0) dot = 0;
    __syncthreads();

    for(int i = 0; i < loopTimes; ++i){
        int idx = i * BLOCK_THREADS + tx;

        scalar_t prod = (idx < length)? a[idx] * b[idx] : (scalar_t) 0;
        scalar_t reduce = BlockReduce(tmpStorage).Sum(prod);

        if(tx == 0) dot += reduce;
        __syncthreads();
    }

    return dot;
}

template <int BLOCK_THREADS, typename scalar_t> 
__global__ void reflections(scalar_t *R, scalar_t *vs, int m, int n, semaphore *sems){ //vs always float precision?
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int vLen = n - bx;
    scalar_t *v = &vs[bx * n + bx];

    if(tx == 0)
        sems[bx * m + bx].acquire();
    __syncthreads();

    for(int idx = tx; idx < vLen; idx += BLOCK_THREADS)
        v[idx] = - R[bx * n + bx + idx];

    scalar_t normVSq = dot<BLOCK_THREADS, scalar_t>(v, v, vLen, tx);
    if(tx == 0) v[0] += copysign(sqrt(normVSq), v[0]);
    
    scalar_t normV = sqrt(dot<BLOCK_THREADS, scalar_t>(v, v, vLen, tx));
    for(int idx = tx; idx < vLen; idx += BLOCK_THREADS)
        v[idx] /= normV;

    for(int row = bx + 1; row < m; ++row){
        if(tx == 0) sems[bx * m + row].acquire();   
        __syncthreads();
        
        scalar_t dotValue = dot<BLOCK_THREADS, scalar_t>(&R[row * n + bx], v, vLen, tx);
        
        for(int idx = tx; idx < vLen; idx += BLOCK_THREADS)
            R[row * n + bx + idx] -= 2.0 * v[idx] * dotValue;

        __syncthreads();
        if (tx == 0) sems[(bx + 1) * m + row].release();
    }
}

template <int BLOCK_THREADS, typename scalar_t> 
__global__  void QLoop(scalar_t *Q, scalar_t *vs, int n, int m){
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    for(int vIdx = 0; vIdx < m; ++vIdx){
        scalar_t *v = &vs[vIdx * n];
    
        scalar_t dotValue = dot<BLOCK_THREADS, scalar_t>(v, &Q[bx * n], n, tx);

        for(int idx = tx; idx < n; idx += BLOCK_THREADS)
            Q[bx * n + idx] -= 2.0 * v[idx] * dotValue;
    }
}

__global__ 
void initSems(semaphore *sems, int m){
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    new (&sems[bx * m + tx]) semaphore (bx == 0);
}

template <typename scalar_t> 
void dispatchedImplementation(torch::Tensor A, int m, int n, float epsilon){
    semaphore *sems;
    cudaMalloc((void**)&sems, (m + 1) * m * sizeof(semaphore));
    initSems<<<m + 1, m>>>(sems, m);
    
    torch::Tensor vs = torch::zeros_like(A);
    A.diagonal().add_((scalar_t) epsilon);
    
    cudaDeviceSynchronize();
    reflections<BLOCK_THREADS, scalar_t><<<m, BLOCK_THREADS>>>(A.data<scalar_t>(), vs.data<scalar_t>(), m, n, sems);
    cudaDeviceSynchronize();

    cudaMemset(A.data<scalar_t>(), 0, m * n * sizeof(scalar_t));
    A.fill_diagonal_(1);

    cudaDeviceSynchronize();

    QLoop<BLOCK_THREADS, scalar_t><<<m, BLOCK_THREADS>>>(A.data<scalar_t>(), vs.data<scalar_t>(), n, m);
    cudaDeviceSynchronize();

    cudaFree(sems);
}

void qrOrthogonalizationCuda(torch::Tensor A, int m, int n, float epsilon){
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
    A.scalar_type(), "qr_orthogonalization_cuda", ([&] {
        dispatchedImplementation<scalar_t>(A, m, n, epsilon);
    }));
}
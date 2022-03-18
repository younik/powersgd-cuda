#include <cub/cub.cuh>
#include <cuda/std/semaphore>
#include <torch/extension.h>

using semaphore = cuda::std::counting_semaphore<>;
const int BLOCK_THREADS = 512;


template <int BLOCK_THREADS, typename scalar_t>
__device__  scalar_t dot(scalar_t *a, scalar_t *b, int length, int tx){
    typedef cub::BlockReduce<scalar_t, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int loop_times = ceil((float)length / (float)BLOCK_THREADS);
    __shared__ scalar_t dot;
    if(tx == 0) dot = 0;
    __syncthreads();

    for(int i = 0; i < loop_times; ++i){
        int idx = i * BLOCK_THREADS + tx;
        scalar_t prod = 0;
        if(idx < length) prod = a[idx] * b[idx];
        
        scalar_t reduce = BlockReduce(temp_storage).Sum(prod);

        if(tx == 0) dot += reduce;
        __syncthreads();
    }

    return dot;
}

template <int BLOCK_THREADS, typename scalar_t> 
__global__ void reflections(scalar_t *R, scalar_t *vs, int m, int n, semaphore *sems){ //vs still float precision?
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int v_len = n - bx;
    scalar_t *v = &vs[bx * n + bx];

    if(tx == 0)
        sems[bx * m + bx].acquire();
    __syncthreads();

    for(int idx = tx; idx < v_len; idx += BLOCK_THREADS)
        v[idx] = - R[bx * n + bx + idx];

    scalar_t norm_v_sq = dot<BLOCK_THREADS, scalar_t>(v, v, v_len, tx);
    if(tx == 0) v[0] += copysign(sqrt(norm_v_sq), v[0]);
    
    scalar_t norm_v = sqrt(dot<BLOCK_THREADS, scalar_t>(v, v, v_len, tx));
    for(int idx = tx; idx < v_len; idx += BLOCK_THREADS)
        v[idx] /= norm_v;

    for(int row = 0; row < m; ++row){ //dynamic parallelsim and avoid this loop?
        if(row > bx){
            if(tx == 0) sems[bx * m + row].acquire();   
            __syncthreads();
        }     

        scalar_t dot_value = dot<BLOCK_THREADS, scalar_t>(&R[row * n + bx], v, v_len, tx);
        
        for(int idx = tx; idx < v_len; idx += BLOCK_THREADS)
            R[row * n + bx + idx] -= 2.0 * v[idx] * dot_value;

        if(row > bx){
            __syncthreads();
            if (tx == 0) sems[(bx + 1) * m + row].release();
        }
    }
}

template <int BLOCK_THREADS, typename scalar_t> 
__global__  void Q_loop(scalar_t *Q, scalar_t *vs, int n, int m, semaphore *sems){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int v_idx = m - blockIdx.y - 1;
    scalar_t *v = &vs[v_idx * n];

    if(tx==0) sems[(v_idx + 1) * m + bx].acquire();
    __syncthreads();
    
    scalar_t dot_value = dot<BLOCK_THREADS, scalar_t>(v, &Q[bx * n], n, tx);

    for(int idx = tx; idx < n; idx += BLOCK_THREADS)
        Q[bx * n + idx] -= 2.0 * v[idx] * dot_value;

    __syncthreads();
    if(tx==0) sems[v_idx  * m + bx].release();
}

template <typename scalar_t> 
__global__ void add_diag(scalar_t *A, int n, scalar_t value){
    int tx = threadIdx.x;
    A[tx * n + tx] += value;
}

__global__ 
void init_sems(semaphore *sems, int m){
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    new (&sems[bx * m + tx]) semaphore (bx == 0);
}

__global__
void release_sems(semaphore *sems){
    sems[threadIdx.x].release();
}

template <typename scalar_t> 
void dispatched_implementation(torch::Tensor A, int m, int n, float epsilon){
    semaphore *sems;
    cudaMalloc((void**)&sems, (m + 1) * m * sizeof(semaphore));
    init_sems<<<m + 1, m>>>(sems, m);
    
    scalar_t *vs;
    cudaMalloc(&vs, m * n * sizeof(scalar_t));
    cudaMemsetAsync(vs, 0, m * n * sizeof(scalar_t));

    scalar_t eps = (scalar_t) epsilon;
    add_diag<scalar_t><<<1, m>>>(A.data<scalar_t>(), n, eps);

    cudaDeviceSynchronize();
    reflections<BLOCK_THREADS, scalar_t><<<m, BLOCK_THREADS>>>(A.data<scalar_t>(), vs, m, n, sems);
    cudaDeviceSynchronize();

    release_sems<<<1, m>>>(&sems[m*m]);
    cudaMemset(A.data<scalar_t>(), 0, m * n * sizeof(scalar_t));
    add_diag<scalar_t><<<1, m>>>(A.data<scalar_t>(), n, 1);
    
    cudaDeviceSynchronize();

    dim3 blockDim = dim3(m, m);
    Q_loop<BLOCK_THREADS, scalar_t><<<blockDim, BLOCK_THREADS>>>(A.data<scalar_t>(), vs, n, m, sems);
    cudaDeviceSynchronize();

    cudaFree(sems);
    cudaFree(vs);
}

void qr_orthogonalization_cuda(torch::Tensor A, int m, int n, float epsilon){
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
    A.scalar_type(), "qr_orthogonalization_cuda", ([&] {
        dispatched_implementation<scalar_t>(A, m, n, epsilon);
    }));
}
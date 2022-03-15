#include <vector>
#include <cub/cub.cuh>
#include <cuda/std/semaphore>
#include <torch/extension.h>

using semaphore = cuda::std::counting_semaphore<>;


template <int BLOCK_THREADS, typename scalar_t> 
__global__ void reflections(scalar_t *R, scalar_t *vs, int m, int n, semaphore *sems){ //vs still double precision?
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int v_len = n - bx;
    __shared__ scalar_t *v = new scalar_t[v_len];
    int m_pos = bx * n + tx;

    if(tx == 0)
        sems[bx * m + bx].acquire();
    __syncthreads();

    for(int idx = tx; idx < v_len; idx += BLOCK_THREADS)
        v[idx] = - R[bx * n + bx + idx];

    scalar_t norm_v = sqrt(dot<BLOCK_THREADS, scalar_t>(v, v, v_len, tx));
    if(tx == 0) v[0] += copysign(norm_v, v[0]);

    norm_v = sqrt(dot<BLOCK_THREADS, scalar_t>(v, v, v_len, tx));
    for(int idx = tx; idx < v_len; idx += BLOCK_THREADS)
        v[idx] /= norm_v;

    for(int row = 0; row < m; ++row){ //dynamic parallelsim and avoid this loop?
        if(row > bx){
            if(tx == 0)
                sems[bx * m + row].acquire();   
            __syncthreads();
        }     

        scalar_t prod = R[row * n + tx] * vs[m_pos];
        scalar_t dot = dot<BLOCK_THREADS, scalar_t>(&R[row * n + bx], v, v_len, tx);
        
        for(int idx = tx; idx < v_len; idx += BLOCK_THREADS)
            R[row * n + bx + idx] -= 2.0 * v[idx] * dot;

        if(row > bx){
            __syncthreads();
            if (tx == 0) sems[(bx + 1) * m + row].release();
        }
    }

    for(int idx = tx; idx < v_len; idx += BLOCK_THREADS)
        vs[bx * m + bx + idx] = v[idx]; //TODO: vs not matrix, variable row size
}

template <int BLOCK_THREADS, typename scalar_t> 
__global__  scalar_t dot(scalar_t *a, scalat_t *b, int length, int tx){
    typedef cub::BlockReduce<scalar_t, BLOCK_THREADS, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    __shared__ scalar_t dot = 0;
    for(int idx = tx; idx < length; idx += BLOCK_THREADS){
        scalar_t prod = a[idx] * b[idx];
        scalar_t reduce = BlockReduce(temp_storage).Sum(prod); //not all thread calls when length not power of 2, ok?

        if(tx == 0) dot += reduce;
        __syncthreads();
    }

    return dot;
}

template <int BLOCK_THREADS, typename scalar_t> 
__global__  void Q_loop(scalar_t *Q, scalar_t *vs, int n, int col){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    
    scalar_t dot = dot<BLOCK_THREADS, scalar_t>(&vs[col * n], &Q[bx * n], n, tx);

    for(int idx = tx; idx < n; idx += BLOCK_THREADS)
        Q[bx * n + idx] -= 2.0 * vs[col * n + idx] * dot;
}

template <typename scalar_t> 
__global__ void add_diag(scalar_t *A, int n, scalar_t value){
    int tx = threadIdx.x;
    A[tx * n + tx] += value;
}


template <typename scalar_t> 
void dispatched_implementation(torch::Tensor A, torch::Tensor Q, int m, int n, float epsilon){
    scalar_t *vs; //device
    scalar_t eps = (scalar_t) epsilon;
    
    add_diag<scalar_t><<<1, m>>>(A.data<scalar_t>(), n, eps);
    cudaMalloc(&vs, m*n*sizeof(scalar_t));

    semaphore *sems;
    cudaMalloc(&sems, m*(m+1)*sizeof(semaphore));
    for(int i=0; i<m; ++i){ //init on device?
        semaphore sem_h(1);
        cudaMemcpyAsync(&sems[i], &sem_h, sizeof(semaphore), cudaMemcpyHostToDevice);
    }
    for(int i=m; i<m*(m+1); ++i){
        semaphore sem_h(0);
        cudaMemcpyAsync(&sems[i], &sem_h, sizeof(semaphore), cudaMemcpyHostToDevice);
    }

    cudaDeviceSynchronize();

    reflections<1024, scalar_t><<<m, n>>>(A.data<scalar_t>(), vs, m, n, sems);
    
    cudaMemset(Q.data<scalar_t>(), 0, m * n * sizeof(scalar_t));
    add_diag<scalar_t><<<1, m>>>(Q.data<scalar_t>(), n, 1);
    cudaDeviceSynchronize();

    for(int col = m - 1; col >= 0; --col){
        Q_loop<1024, scalar_t><<<m, n>>>(Q.data<scalar_t>(), vs, n, col);
        cudaDeviceSynchronize();
    }

    cudaFree(sems);
    cudaFree(vs);
}

void qr_orthogonalization_cuda(torch::Tensor A, torch::Tensor Q, int m, int n, float epsilon){
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
    A.scalar_type(), "qr_orthogonalization_cuda", ([&] {
        dispatched_implementation<scalar_t>(A, Q, m, n, epsilon);
    }));
}

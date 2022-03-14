#include <vector>
#include <cub/cub.cuh>
#include <cuda/std/semaphore>
#include <torch/extension.h>

using semaphore = cuda::std::counting_semaphore<>;


template <int N, typename scalar_t> 
__global__ void reflections(scalar_t *R, scalar_t *vs, int m, semaphore *sems){ //vs still double precision?
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int m_pos = bx * N + tx;
    typedef cub::BlockReduce<scalar_t, N, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    if(tx == 0)
        sems[bx * m + bx].acquire();
    __syncthreads();

    vs[m_pos] = - (tx >= bx) * R[m_pos]; //try vs[m_pos] in shared mem and push in global at the end

    scalar_t z_sq = pow(vs[m_pos], 2);
    scalar_t norm_z_sq = BlockReduce(temp_storage).Sum(z_sq);
    if(tx == bx)
        vs[m_pos] -= copysign(sqrt(norm_z_sq), R[m_pos]);

    scalar_t v_sq = pow(vs[m_pos], 2); 
    scalar_t norm_v_sq = BlockReduce(temp_storage).Sum(v_sq);
    __shared__ scalar_t norm_v;
    if(tx == 0)
        norm_v = sqrt(norm_v_sq);
    __syncthreads();

    vs[m_pos] /= norm_v;

    for(int row = 0; row < m; ++row){ //dynamic parallelsim and avoid this loop?
        if(row > bx){
            if(tx == 0)
                sems[bx * m + row].acquire();   
            __syncthreads();
        }     

        scalar_t prod = R[row * N + tx] * vs[m_pos];
        scalar_t reduce = BlockReduce(temp_storage).Sum(prod);
        __shared__ scalar_t dot;
        if(tx == 0)
           dot = reduce;
        __syncthreads();

        R[m * N + tx] -= 2.0 * vs[m_pos] * dot;

        //if(m > bx){
        __syncthreads();
        if (tx == 0)
            sems[(bx + 1) * m + row].release(); //possible also with if(tx==0) .release(N)
        //}
    }
}

template <int N, typename scalar_t> 
__global__  void Q_loop(scalar_t *Q, scalar_t *vs, int col){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    typedef cub::BlockReduce<scalar_t, N, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    scalar_t prod = vs[col * N + tx] * Q[bx * N + tx];
    scalar_t reduce = BlockReduce(temp_storage).Sum(prod);
    __shared__ scalar_t dot;
    if(tx == 0)
       dot = reduce;
    __syncthreads();

    Q[bx * N + tx] -= 2.0 * vs[col * N + tx] * dot;
}

template <typename scalar_t> 
__global__ void add_diag(scalar_t *A, int n, scalar_t value){
    int tx = threadIdx.x;
    A[tx * n + tx] += value;
}


template <typename scalar_t> 
void dispatched_implementation(torch::Tensor A, torch::Tensor Q, int m, const int n, float epsilon){
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
    reflections<1024, scalar_t><<<m, n>>>(A.data<scalar_t>(), vs, m, sems);
    
    cudaMemset(Q.data<scalar_t>(), 0, m * n * sizeof(scalar_t));
    add_diag<scalar_t><<<1, m>>>(Q.data<scalar_t>(), n, 1);
    cudaDeviceSynchronize();

    for(int col = m - 1; col >= 0; --col){
        Q_loop<1024, scalar_t><<<m, n>>>(Q.data<scalar_t>(), vs, col);
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

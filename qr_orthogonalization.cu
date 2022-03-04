#include <vector>
#include <cub/cub.cuh>
#include <cuda/std/semaphore>
#include <torch/extension.h>

using semaphore = cuda::std::counting_semaphore<>;

const int M = 8;
const int N = 1024;


__global__ 
void reflections(float *R, float *vs, semaphore *sems){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int m_pos = bx * N + tx;
    typedef cub::BlockReduce<float, N, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    if(tx == 0)
        sems[bx * M + bx].acquire(); //warp accessing simultanuosly? not good
    __syncthreads();

    vs[m_pos] = - (tx >= bx) * R[m_pos]; //try vs[m_pos] in shared mem and push in global at the end

    float z_sq = pow(vs[m_pos], 2);
    float norm_z_sq = BlockReduce(temp_storage).Sum(z_sq);
    if(tx == bx)
        vs[m_pos] -= copysign(sqrt(norm_z_sq), R[m_pos]);

    float v_sq = pow(vs[m_pos], 2); 
    float norm_v_sq = BlockReduce(temp_storage).Sum(v_sq);
    __shared__ float norm_v;
    if(tx == 0)
        norm_v = sqrt(norm_v_sq);
    __syncthreads();

    vs[m_pos] /= norm_v;

    for(int m = 0; m < M; ++m){ //dynamic parallelsim and avoid this loop?
        if(m > bx){
            if(tx == 0)
                sems[bx * M + m].acquire();   
            __syncthreads();
        }     

        float prod = R[m * N + tx] * vs[m_pos];
        float reduce = BlockReduce(temp_storage).Sum(prod);
        __shared__ float dot;
        if(tx == 0)
           dot = reduce;
        __syncthreads();

        R[m * N + tx] -= 2.0 * vs[m_pos] * dot;

        //if(m > bx){
        __syncthreads();
        if (tx == 0)
            sems[(bx + 1) * M + m].release(); //possible also with if(tx==0) .release(N)
        //}
    }
}

__global__ 
void Q_loop(float *Q, float *vs, int col){
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    typedef cub::BlockReduce<float, N, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float prod = vs[col * N + tx] * Q[bx * N + tx];
    float reduce = BlockReduce(temp_storage).Sum(prod);
    __shared__ float dot;
    if(tx == 0)
       dot = reduce;
    __syncthreads();

    Q[bx * N + tx] -= 2.0 * vs[col * N + tx] * dot;
}

__global__
void Q_init(float *Q){
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    Q[bx * N + tx] = (bx == tx);
}

__global__
void add_eps_diag(float *R, int N, float epsilon){
    int tx = threadIdx.x;
    R[tx * N + tx] += epsilon;
}

void qr_orthogonalization_cuda(torch::Tensor A, torch::Tensor Q, int M, int N, float epsilon){
    float *R, *vs; //device
    
    R = A.data<float>();
    add_eps_diag<<<1, M>>>(R, N, epsilon);
    cudaMalloc(&vs, M*N*sizeof(float));

    semaphore *sems;
    cudaMalloc(&sems, M*(M+1)*sizeof(semaphore));
    for(int i=0; i<M; ++i){ //init on device?
        semaphore sem_h(N);
        cudaMemcpyAsync(&sems[i], &sem_h, sizeof(semaphore), cudaMemcpyHostToDevice);
    }
    for(int i=M; i<M*(M+1); ++i){
        semaphore sem_h(0);
        cudaMemcpyAsync(&sems[i], &sem_h, sizeof(semaphore), cudaMemcpyHostToDevice);
    }

    cudaDeviceSynchronize();

    //for(int i=0; i<M; ++i)
    reflections<<<M, N>>>(R, vs, sems);
    
    // // cudaDeviceSynchronize();
    Q_init<<<M, N>>>(Q.data<float>()); //TODO: float change with template
    //try memset
    // //cudaDeviceSynchronize();

    for(int col = M - 1; col >= 0; --col){
        Q_loop<<<M, N>>>(Q.data<float>(), vs, col);
        //cudaDeviceSynchronize();
    }

    cudaFree(sems);
    cudaFree(vs);
}

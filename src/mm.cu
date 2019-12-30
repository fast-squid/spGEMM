#include <stdio.h>
#include "../lib/coo.h"
#include "../lib/cm.h"
#include "../lib/mm.h"

#define ERROR_CHECK \
{\
	cudaError err = cudaGetLastError(); \
	if ( cudaSuccess != err ) \
	{\
		printf("[%s:%d]CUDA ERROR : %s\n", __FILE__, __LINE__, cudaGetErrorString(err) ); \
		exit(-1); \
	}\
}

int* a_ptr;
int* a_idx;
float* a_val;

int* b_ptr;
int* b_idx;
float* b_val;

int* c_ptr_nnz;
int* c_ptr_base;
int* c_idx;
float* c_val;


__device__ int a_num_cols, a_num_rows;
__device__ bool a_type;
__device__ int b_num_cols, b_num_rows;
__device__ bool b_type;

__global__ void coo2cm(/*{{{*/
        triplet* data, int* nnz, bool* type,
        int* ptr, int *idx, float* val)
{
    int global_tid = threadIdx.x+blockDim.x*blockIdx.x;
    if(global_tid<*nnz)
    {
        if(*type == ROW_MAJOR)
        {
            idx[global_tid] = data[global_tid].cidx;
            val[global_tid] = data[global_tid].val;
            atomicAdd(&ptr[data[global_tid].ridx+1], 1);
        }
        else
        {
            idx[global_tid] = data[global_tid].ridx;
            val[global_tid] = data[global_tid].val;
            atomicAdd(&ptr[data[global_tid].cidx+1], 1);
        }
    }
}

cm cudaCOO2CM(coo mat)
{
    cm ret;
    cmInit(&ret);
    
    cmSetNNZ(&ret, mat.nnz);
    cmSetNumRows(&ret, mat.num_rows);
    cmSetNumCols(&ret, mat.num_cols);
    cmSetType(&ret, mat.type);

    triplet *data;
    bool *type;
    int *nnz;

    cudaMalloc((void**)&data, sizeof(triplet)*mat.nnz);
    cudaMalloc((void**)&type, sizeof(bool));
    cudaMalloc((void**)&nnz, sizeof(int));
    cudaMemcpy(data, mat.data, sizeof(triplet)*mat.nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(type, &mat.type, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(nnz, &mat.nnz, sizeof(int), cudaMemcpyHostToDevice);

    // CSC
    if(mat.type==COL_MAJOR)
    {
        cudaMalloc((void**)&a_ptr, sizeof(int)*(ret.num_rows+1));
        cudaMalloc((void**)&a_idx, sizeof(int)*(ret.nnz));
        cudaMalloc((void**)&a_val, sizeof(float)*(ret.nnz));

        coo2cm<<< mat.nnz/128+1, 128>>>
            (data, nnz, type,
             a_ptr, a_idx, a_val);

        ret.ptr = new int[ret.num_rows+1];
        ret.idx = new int[ret.nnz];
        ret.val = new float[ret.nnz];
        cudaMemcpy(ret.ptr, a_ptr, sizeof(int)*(ret.num_rows+1),cudaMemcpyDeviceToHost);
        cudaMemcpy(ret.idx, a_idx, sizeof(int)*ret.nnz,cudaMemcpyDeviceToHost);
        cudaMemcpy(ret.val, a_val, sizeof(float)*ret.nnz,cudaMemcpyDeviceToHost);

        ret.ptr[0] = 0;
        for(int i=0; i<ret.num_rows;i++){
            ret.ptr[i+1] += ret.ptr[i];
        }
        cudaMemcpy(a_ptr, ret.ptr, sizeof(int)*(ret.num_rows+1),cudaMemcpyHostToDevice);
    }
    else{
        cudaMalloc((void**)&b_ptr, sizeof(int)*(ret.num_cols+1));
        cudaMalloc((void**)&b_idx, sizeof(int)*(ret.nnz));
        cudaMalloc((void**)&b_val, sizeof(float)*(ret.nnz));

        coo2cm<<< mat.nnz/128+1, 128>>>
            (data,nnz,type,
             b_ptr, b_idx, b_val);

        ret.ptr = new int[ret.num_cols+1];
        ret.idx = new int[ret.nnz];
        ret.val = new float[ret.nnz];
        cudaMemcpy(ret.ptr, b_ptr, sizeof(int)*(ret.num_cols+1),cudaMemcpyDeviceToHost);
        cudaMemcpy(ret.idx, b_idx, sizeof(int)*ret.nnz,cudaMemcpyDeviceToHost);
        cudaMemcpy(ret.val, b_val, sizeof(float)*ret.nnz,cudaMemcpyDeviceToHost);

        ret.ptr[0] = 0;
        for(int i=0; i<ret.num_cols;i++){
            ret.ptr[i+1] += ret.ptr[i];
        }
        cudaMemcpy(b_ptr, ret.ptr, sizeof(int)*(ret.num_cols+1),cudaMemcpyHostToDevice);
    }
    cudaFree(data);
    cudaFree(type);
    return ret;
}/*}}}*/

__global__ void initGEMM(
        int* a_ptr, int* a_idx,
        int* b_ptr, int* b_idx,
        int *c_ptr)
{
    for(int i = a_ptr[blockIdx.x] + threadIdx.x; i < a_ptr[blockIdx.x+1]; i+= blockDim.x)
    {
        int a_ridx= a_idx[i];
        atomicAdd(&c_ptr[a_ridx], b_ptr[blockIdx.x+1] - b_ptr[blockIdx.x]);
    }
}

cm cudaInitGEMM(cm A, cm B)
{
    cm C;
    cmSetType(&C, ROW_MAJOR);
    cmSetNumRows(&C, cmGetNumRows(A));
    cmSetNumCols(&C, cmGetNumCols(B));

    cudaMemcpyToSymbol(a_num_cols, &A.num_cols, sizeof(int),0 ,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(a_num_rows, &A.num_rows, sizeof(int),0 ,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(b_num_cols, &B.num_cols, sizeof(int),0 ,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(b_num_rows, &B.num_rows, sizeof(int),0 ,cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&c_ptr_base,sizeof(int)*cmGetNumRows(C)+1);
    cudaMalloc((void**)&c_ptr_nnz,sizeof(int)*cmGetNumRows(C)+1);

    initGEMM<<< A.num_cols, 32 >>>
        (a_ptr, a_idx,
         b_ptr, b_idx,
         c_ptr_base);
    
    int *temp = new int[A.num_rows+1];
    cudaMemcpy(temp, c_ptr_base, sizeof(int)*(A.num_rows+1), cudaMemcpyDeviceToHost);
    for(int i=0;i<A.num_rows;i++)
    {
        temp[i+1] += temp[i];
    }
    cmSetNNZ(&C,temp[A.num_rows]);
    printf("%d\n",cmGetNNZ(C));
    
    cudaMalloc((void**)&c_idx, sizeof(int)*temp[A.num_rows]);
    cudaMalloc((void**)&c_val, sizeof(int)*temp[A.num_rows]);
    cudaMemcpy(c_ptr_base, temp,sizeof(int)*(A.num_rows+1), cudaMemcpyHostToDevice);
    delete temp;
    return C;
}


__global__ void simpleGEMM(
        int* a_ptr, int* a_idx, float*a_val,
        int* b_ptr, int* b_idx, float*b_val,
        int* c_ptr_base, int* c_ptr_nnz, int* c_idx, float* c_val)
{
    __shared__ int offset;
    for(int ai = a_ptr[blockIdx.x]; ai < a_ptr[blockIdx.x+1]; ai++)
    {
        if(threadIdx.x==0)
            offset = atomicAdd(&c_ptr_nnz[blockIdx.x], b_ptr[blockIdx.x+1] - b_ptr[blockIdx.x]);
        __syncthreads();
        int base = c_ptr_base[blockIdx.x];
        for(int bi = threadIdx.x; bi < b_ptr[blockIdx.x+1] - b_ptr[blockIdx.x]; bi+=blockDim.x)
        {
            c_val[base+offset+bi] = a_val[ai]*b_val[bi];
            c_idx[base+offset+bi] = b_idx[bi];
        }
    }
}

void cudaGEMM(cm A, cm B, cm C)
{
    simpleGEMM<<<cmGetNumCols(A), 128 >>>
        (a_ptr, a_idx, a_val,
         b_ptr, b_idx, b_val,
         c_ptr_base, c_ptr_nnz , c_idx, c_val);

}

/*
__global__ void gemm(
        int* a_ptr, int* a_idx, float* a_val,
        int* b_ptr, int* b_idx, float* b_val,
        int* c_ptr, int* c_idx, float* c_val,
        )
{
    int idx = blockIdx.x;

    for(int a_iter = a_ptr[blockIdx.x]; a_iter < a_ptr[blockIdx.x+1]; a_iter++)
    {
        if(threadIdx.x==0)
            atomicAdd(&c_ptr[], b_ptr[blockIdx.x+1] - b_ptr[blockIdx.x]);        
        __syncthreads();
        for(int b_iter = b_ptr[blockIdx.x].ptr; b_iter < b_ptr[blockIdx.x+1]; b_iter += blockDim.x)
        {
            // row = aidx;
            // col = bidx;
            // val = 
            c_val[ + b_iter] = a_val[a_iter]*b_val[b_iter];
            c_idx[ + b_iter] = b_idx[b_iter];
        }
    }
}
*/
/*
void test(cm A, cm B)
{
    int nnzC=0;
    int number_of_ops = A.num_cols;
    printf("test\n");
    int* visit = new int[A.num_cols]; 

    for(int i=0;i<A.num_cols;i++){
        int k = i;
        for(int j=1; j<=B.ptr[k+1] - B.ptr[k];j++){ 
            if(visit[k] == 0){
                visit[k] = 1;
                int a = (A.ptr[k+1] - A.ptr[k]);
                int b = (B.ptr[k+1] - B.ptr[k]);
                int temp = a*b;
                nnzC+=temp;
                printf("%d 번째 %d %d = %d\n",k,a,b,a*b);
            }
            else{
                printf("%d skip\n",k);
            }
            if(B.ptr[k] + j<B.nnz)
            k=B.idx[B.ptr[k]+j]; 
        }
        printf("다음\n");
    }

    printf("total ops : %d\n",nnzC);
}
*/

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


__device__ int a_row_size, a_col_size;
__device__ bool a_type;
__device__ int b_row_size, b_col_size;
__device__ bool b_type;

__global__ void coo2cm(
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

cm cudaCOO2CM(coo mat){/*{{{*/
    cm ret;
    initCM(&ret);

    ret.nnz = mat.nnz;
    ret.row_size = mat.row_size;
    ret.col_size = mat.col_size;
    ret.type = mat.type;
    setCMtype(&ret, mat.type);

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
        cudaMalloc((void**)&a_ptr, sizeof(int)*(ret.col_size+1));
        cudaMalloc((void**)&a_idx, sizeof(int)*(ret.nnz));
        cudaMalloc((void**)&a_val, sizeof(float)*(ret.nnz));

        coo2cm<<< mat.nnz/128+1, 128>>>
            (data, nnz, type,
             a_ptr, a_idx, a_val);

        ret.ptr = new int[ret.col_size+1];
        ret.idx = new int[ret.nnz];
        ret.val = new float[ret.nnz];
        cudaMemcpy(ret.ptr, a_ptr, sizeof(int)*(ret.col_size+1),cudaMemcpyDeviceToHost);
        cudaMemcpy(ret.idx, a_idx, sizeof(int)*ret.nnz,cudaMemcpyDeviceToHost);
        cudaMemcpy(ret.val, a_val, sizeof(float)*ret.nnz,cudaMemcpyDeviceToHost);

        ret.ptr[0] = 0;
        for(int i=0; i<ret.col_size;i++){
            ret.ptr[i+1] += ret.ptr[i];
        }
        cudaMemcpy(a_ptr, ret.ptr, sizeof(int)*(ret.col_size+1),cudaMemcpyHostToDevice);
    }
    else{
        cudaMalloc((void**)&b_ptr, sizeof(int)*(ret.row_size+1));
        cudaMalloc((void**)&b_idx, sizeof(int)*(ret.nnz));
        cudaMalloc((void**)&b_val, sizeof(float)*(ret.nnz));

        coo2cm<<< mat.nnz/128+1, 128>>>
            (data,nnz,type,
             b_ptr, b_idx, b_val);

        ret.ptr = new int[ret.row_size+1];
        ret.idx = new int[ret.nnz];
        ret.val = new float[ret.nnz];
        cudaMemcpy(ret.ptr, b_ptr, sizeof(int)*(ret.row_size+1),cudaMemcpyDeviceToHost);
        cudaMemcpy(ret.idx, b_idx, sizeof(int)*ret.nnz,cudaMemcpyDeviceToHost);
        cudaMemcpy(ret.val, b_val, sizeof(float)*ret.nnz,cudaMemcpyDeviceToHost);

        ret.ptr[0] = 0;
        for(int i=0; i<ret.row_size;i++){
            ret.ptr[i+1] += ret.ptr[i];
        }
        cudaMemcpy(b_ptr, ret.ptr, sizeof(int)*(ret.row_size+1),cudaMemcpyHostToDevice);
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

    for(int i = b_ptr[blockIdx.x] + threadIdx.x; i< b_ptr[blockIdx.x+1] ; i += blockDim.x)
    {
        int b_ridx = b_idx[i];
        atomicAdd(&c_ptr[blockIdx.x+1], b_ptr[b_ridx+1] - b_ptr[b_ridx]);
    }
}

void cudaInitGEMM(cm A, cm B)
{
    
    cudaMemcpyToSymbol(a_row_size, &A.row_size, sizeof(int),0 ,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(a_col_size, &A.col_size, sizeof(int),0 ,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(b_row_size, &B.row_size, sizeof(int),0 ,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(b_col_size, &B.col_size, sizeof(int),0 ,cudaMemcpyHostToDevice);

    
    cudaMalloc((void**)&c_ptr_base,sizeof(int)*(A.col_size+1));
    initGEMM<<< A.col_size, 32 >>>
        (a_ptr, a_idx, b_ptr, b_idx, c_ptr_base);
    
    int *temp = new int[A.col_size+1];
    cudaMemcpy(temp, c_ptr_base, sizeof(int)*(A.col_size+1), cudaMemcpyDeviceToHost);
    for(int i=0;i<A.col_size;i++)
    {
        temp[i+1] += temp[i];
    }
    printf("%d",temp[A.col_size]);
    
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
    int number_of_ops = A.row_size;
    printf("test\n");
    int* visit = new int[A.row_size]; 

    for(int i=0;i<A.row_size;i++){
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

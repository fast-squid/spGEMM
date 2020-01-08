#include <stdio.h>
#include "../lib/coo.h"
#include "../lib/cm.h"
#include "../lib/mm.h"
/*{{{*/
#define ERROR_CHECK \
{\
	cudaError err = cudaGetLastError(); \
	if ( cudaSuccess != err ) \
	{\
		printf("[%s:%d]CUDA ERROR : %s\n", __FILE__, __LINE__, cudaGetErrorString(err) ); \
		exit(-1); \
	}\
}
/*}}}*/

#define DENSE_NUM 1000

int* match;
int* num_splits;

int* a_ptr;
int* a_idx;
float* a_val;

int* b_ptr;
int* b_idx;
float* b_val;

int* c_ptr_nnz;
int* c_ptr_base;
int* c_idx;
int* c_idx_dummy;
float* c_val;

int* dense;


__device__ int a_num_cols, a_num_rows;
__device__ bool a_type;
__device__ int b_num_cols, b_num_rows;
__device__ bool b_type;
__device__ int c_num_cols, c_num_rows, c_nnz;
__device__ bool c_type;

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
        cudaMalloc((void**)&a_ptr, sizeof(int)*(ret.num_cols+1));
        cudaMalloc((void**)&a_idx, sizeof(int)*(ret.nnz));
        cudaMalloc((void**)&a_val, sizeof(float)*(ret.nnz));

        coo2cm<<< mat.nnz/128+1, 128>>>
            (data, nnz, type,
             a_ptr, a_idx, a_val);

        ret.ptr = new int[ret.num_cols+1];
        ret.idx = new int[ret.nnz];
        ret.val = new float[ret.nnz];
        cudaMemcpy(ret.ptr, a_ptr, sizeof(int)*(ret.num_cols+1),cudaMemcpyDeviceToHost);
        cudaMemcpy(ret.idx, a_idx, sizeof(int)*ret.nnz,cudaMemcpyDeviceToHost);
        cudaMemcpy(ret.val, a_val, sizeof(float)*ret.nnz,cudaMemcpyDeviceToHost);

        ret.ptr[0] = 0;
        for(int i=0; i<ret.num_cols;i++){
            ret.ptr[i+1] += ret.ptr[i];
        }
        cudaMemcpy(a_ptr, ret.ptr, sizeof(int)*(ret.num_cols+1),cudaMemcpyHostToDevice);
    }
    else{
        cudaMalloc((void**)&b_ptr, sizeof(int)*(ret.num_rows+1));
        cudaMalloc((void**)&b_idx, sizeof(int)*(ret.nnz));
        cudaMalloc((void**)&b_val, sizeof(float)*(ret.nnz));

        coo2cm<<< mat.nnz/128+1, 128>>>
            (data,nnz,type,
             b_ptr, b_idx, b_val);

        ret.ptr = new int[ret.num_rows+1];
        ret.idx = new int[ret.nnz];
        ret.val = new float[ret.nnz];
        cudaMemcpy(ret.ptr, b_ptr, sizeof(int)*(ret.num_rows+1),cudaMemcpyDeviceToHost);
        cudaMemcpy(ret.idx, b_idx, sizeof(int)*ret.nnz,cudaMemcpyDeviceToHost);
        cudaMemcpy(ret.val, b_val, sizeof(float)*ret.nnz,cudaMemcpyDeviceToHost);

        ret.ptr[0] = 0;
        for(int i=0; i<ret.num_rows;i++){
            ret.ptr[i+1] += ret.ptr[i];
        }
        cudaMemcpy(b_ptr, ret.ptr, sizeof(int)*(ret.num_cols+1),cudaMemcpyHostToDevice);
    }
    cudaDeviceSynchronize();
    cudaFree(data);
    cudaFree(type);
    return ret;
}/*}}}*/

__global__ void initGEMM(
        int* a_ptr, int* a_idx,
        int* b_ptr, int* b_idx,
        int *c_ptr_base)
{
    for(int ai = a_ptr[blockIdx.x] + threadIdx.x; ai < a_ptr[blockIdx.x+1]; ai+= blockDim.x)
    {
        int row = a_idx[ai];
        atomicAdd(&c_ptr_base[row+1], b_ptr[blockIdx.x+1] - b_ptr[blockIdx.x]);
    }
}

cm cudaInitGEMM(cm A, cm B)/*{{{*/
{
    cm C;
    cmSetType(&C, ROW_MAJOR);
    cmSetNumRows(&C, cmGetNumRows(A));
    cmSetNumCols(&C, cmGetNumCols(B));

    cudaMemcpyToSymbol(a_num_cols, &A.num_cols, sizeof(int),0 ,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(a_num_rows, &A.num_rows, sizeof(int),0 ,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(b_num_cols, &B.num_cols, sizeof(int),0 ,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(b_num_rows, &B.num_rows, sizeof(int),0 ,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_num_cols, &C.num_cols, sizeof(int),0 ,cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_num_rows, &C.num_rows, sizeof(int),0 ,cudaMemcpyHostToDevice);

    cudaMalloc((void**)&c_ptr_base,sizeof(int)*(cmGetNumRows(C)+1));
    cudaMalloc((void**)&c_ptr_nnz,sizeof(int)*cmGetNumRows(C));
    ERROR_CHECK;
    initGEMM<<< A.num_cols, 32 >>>
        (a_ptr, a_idx,
         b_ptr, b_idx,
         c_ptr_base);
	cudaDeviceSynchronize();

    int *temp = new int[cmGetNumRows(C)+1];
    cudaMemcpy(temp, c_ptr_base, sizeof(int)*(cmGetNumRows(C)+1), cudaMemcpyDeviceToHost);

    for(int i=0;i<A.num_rows;i++)
    {
        temp[i+1] += temp[i];
    }
    cmSetNNZ(&C,temp[A.num_rows]);
    printf("%d %d\n",cmGetNNZ(C),temp[A.num_rows]);
    
    cudaMalloc((void**)&c_idx, sizeof(int)*cmGetNNZ(C));
    cudaMalloc((void**)&c_val, sizeof(float)*cmGetNNZ(C));
    cudaMemcpy(c_ptr_base, temp,sizeof(int)*(cmGetNumRows(C)+1), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_nnz, &C.nnz, sizeof(int),0 ,cudaMemcpyHostToDevice);

    cudaMemset(c_ptr_nnz, 0, sizeof(int)*cmGetNumRows(C));
	cudaDeviceSynchronize();

    delete temp;
    return C;
}
/*}}}*/

__global__ void inspectGEMM(/*{{{*/
        int* a_ptr, int* b_ptr, int* counter)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < a_num_cols){
        int a_len = a_ptr[idx+1] - a_ptr[idx];
        int b_len = b_ptr[idx+1] - b_ptr[idx];
        int workload = a_len * b_len;

        if(workload > c_nnz/a_num_cols*50)
        {
            atomicAdd(&counter[0],1);
        }
        else
        {
            int left = 1;
            int right = 2;
            bool flag = 0;
            for(int i=1;i<8;i++)
            {
                if(left<= b_len && b_len<right)
                {
                    flag = 1;
                    atomicAdd(&counter[i],1);
                }
                left<<=1; 
                right<<=1;
            }
            if(!flag)
            {
                atomicAdd(&counter[8],1);
            }
        }
    }
}

__global__ void categorizeGEMM(
        int* a_ptr, int* b_ptr, 
        int* counter, int* bin)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx<a_num_cols){
        int a_len = a_ptr[idx+1] - a_ptr[idx];
        int b_len = b_ptr[idx+1] - b_ptr[idx];
        int workload = a_len * b_len;

        if(workload > c_nnz/a_num_cols*50)
        {
            int loc = atomicAdd(&counter[0],1);
            bin[loc] = idx;
        }
        else
        {
            int left = 1;
            int right = 2;
            bool flag = 0;
            for(int i=1;i<8;i++)
            {
                if(left<= b_len && b_len<right)
                {
                    int loc = atomicAdd(&counter[i],1);
                    flag = 1;
                    bin[loc] = idx;
                }
                left<<=1; 
                right<<=1;
            }
            if(!flag)
            {
                int loc = atomicAdd(&counter[8],1);
                bin[loc] = idx;
            }

        }
    }
}

int* t_counter;
int* counter;
int* bin;
#define NUM_BINS 9
void cudaCategorizeGEMM(cm A, cm B)
{
    cudaMalloc((void**)&counter, sizeof(int)*(NUM_BINS+1));
    cudaMalloc((void**)&bin, sizeof(int)*cmGetNumCols(A));
    cudaMemset(counter, 0, sizeof(int)*(NUM_BINS+1));

    inspectGEMM<<< cmGetNumCols(A)/32+1, 32 >>>
        (a_ptr, b_ptr, counter);
    
    t_counter=new int[10];

    cudaMemcpy(&t_counter[1], counter, sizeof(int)*NUM_BINS,cudaMemcpyDeviceToHost);

    t_counter[0] = 0;
    for(int i=0; i < 9;i++)
    {
        t_counter[i+1] += t_counter[i];
    }
    
    cudaMemcpy(counter, t_counter, sizeof(int)*(NUM_BINS+1),cudaMemcpyHostToDevice);
    categorizeGEMM<<< cmGetNumCols(A)/32+1, 32>>>
        (a_ptr, b_ptr, counter, bin);
    ERROR_CHECK;
}/*}}}*/

#define WIDTH 4/*{{{*/
__global__ void splitGEMM(
    int* a_ptr, int* a_idx, float* a_val,
    int* b_ptr, int* b_idx, float* b_val,
    int* c_ptr_base, int* c_ptr_nnz, int* c_idx, float* c_val,
    int* match, int* num_splits,
    int*bin)
{
    __shared__ int offset;
    int idx = match[blockIdx.x];
    int bid = blockIdx.x - num_splits[blockIdx.x];

    int b_curr = b_ptr[idx];
    int b_next = b_ptr[idx+1];
    for(int ai = a_ptr[idx]+bid*WIDTH; ai < a_ptr[idx+1] && ai< a_ptr[idx] + (bid+1)*WIDTH; ai++)
    {
        int row = a_idx[ai];
        int base = c_ptr_base[row];
        if(threadIdx.x==0)
            offset = atomicAdd(&c_ptr_nnz[row], b_next-b_curr);
        __syncthreads();
        for(int bi = threadIdx.x; bi < b_next - b_curr; bi+=blockDim.x)
        {
            c_val[base+offset+bi] = a_val[ai]*b_val[b_curr+bi];
            c_idx[base+offset+bi] = b_idx[b_curr+bi];
        }
    }
}

void cudaSplitGEMM(cm A, cm B, cm C)
{   

    int orig = t_counter[1] - t_counter[0];
    int* t_bin = new int[orig];
    int* t_match = new int[cmGetNumRows(A)];
    int* t_num_splits = new int[cmGetNumRows(A)];

    cudaMemcpy(t_bin, bin, sizeof(int)*orig,cudaMemcpyDeviceToHost);
    int num_blocks = 0;
    int prev=0;
    for(int i=0;i<orig;i++)
    {
        int idx = t_bin[i];
        int len =  A.ptr[idx+1] - A.ptr[idx];
        int q = len / WIDTH;
        int r = len % WIDTH;
        int split = r?q+1:q;

        for(int j=0;j<split;j++)
        {
            t_num_splits[num_blocks] = prev;
            t_match[num_blocks++] = idx;
        }
        prev += split;
    }
    cudaMalloc((void**)&match, sizeof(int)*num_blocks);
    cudaMalloc((void**)&num_splits, sizeof(int)*num_blocks);
    cudaMemcpy(match, t_match, sizeof(int)*num_blocks,cudaMemcpyHostToDevice);
    cudaMemcpy(num_splits, t_num_splits, sizeof(int)*num_blocks,cudaMemcpyHostToDevice);
    splitGEMM<<<num_blocks, 32 >>>
            (a_ptr, a_idx, a_val,
             b_ptr, b_idx, b_val,
             c_ptr_base, c_ptr_nnz , c_idx, c_val,
             match, num_splits,
             &bin[t_counter[0]]);

}
/*}}}*/

__global__ void binGEMM(
        int* a_ptr, int* a_idx, float*a_val,
        int* b_ptr, int* b_idx, float*b_val,
        int* c_ptr_base, int* c_ptr_nnz, int* c_idx, float* c_val,
        int* bin)
{
    __shared__ int offset;
    int idx = bin[blockIdx.x];

    int b_curr = b_ptr[idx];
    int b_next = b_ptr[idx+1];
    for(int ai = a_ptr[idx]; ai < a_ptr[idx+1]; ai++)
    {
        int row = a_idx[ai];
        int base = c_ptr_base[row];
        if(threadIdx.x==0)
            offset = atomicAdd(&c_ptr_nnz[row], b_next-b_curr);
        __syncthreads();
        for(int bi = threadIdx.x; bi < b_next - b_curr; bi+=blockDim.x)
        {
            c_val[base+offset+bi] = a_val[ai]*b_val[b_curr+bi];
            c_idx[base+offset+bi] = b_idx[b_curr+bi];
        }
    }
}

void cudaBinGEMM(cm A, cm B, cm C)
{    
    for(int i=1;i<NUM_BINS;i++){
        int num_blocks = t_counter[i+1] - t_counter[i];
        if(num_blocks)
        binGEMM<<<num_blocks, 128 >>>
            (a_ptr, a_idx, a_val,
             b_ptr, b_idx, b_val,
             c_ptr_base, c_ptr_nnz , c_idx, c_val,
             &bin[t_counter[i]]);
    }
    cudaDeviceSynchronize();
    ERROR_CHECK;
}




__global__ void simpleGEMM(/*{{{*/
        int* a_ptr, int* a_idx, float*a_val,
        int* b_ptr, int* b_idx, float*b_val,
        int*c_ptr_base, int* c_ptr_nnz, int* c_idx, float* c_val)
{
    __shared__ int offset;

    int b_curr = b_ptr[blockIdx.x];
    int b_next = b_ptr[blockIdx.x+1];
    for(int ai = a_ptr[blockIdx.x]; ai < a_ptr[blockIdx.x+1]; ai++)
    {
        int row = a_idx[ai];
        int base = c_ptr_base[row];
        if(threadIdx.x==0)
            offset = atomicAdd(&c_ptr_nnz[row], b_next-b_curr);
        __syncthreads();
        for(int bi = threadIdx.x; bi < b_next - b_curr; bi+=blockDim.x)
        {
            c_val[base+offset+bi] = a_val[ai]*b_val[b_curr+bi];
            c_idx[base+offset+bi] = b_idx[b_curr+bi];
        }
    }
}

void cudaSimpleGEMM(cm A, cm B, cm C)
{    

    simpleGEMM<<<cmGetNumCols(A), 32 >>>
        (a_ptr, a_idx, a_val,
         b_ptr, b_idx, b_val,
         c_ptr_base, c_ptr_nnz , c_idx, c_val);
    cudaDeviceSynchronize();
}
/*}}}*/
__global__ void mergeGEMM(/*{{{*/
        int* c_ptr_base, int* c_ptr_nnz, int* c_idx, float* c_val,
        int* c_idx_dummy,int* dense
        )
{

    __shared__ int sh_loc;

    for(int r = 0; r <= c_num_rows/gridDim.x + 1 ; r++)
    {
        __syncthreads();
        sh_loc = 0;
        __syncthreads();
        int bid = (blockIdx.x + r*gridDim.x);
        int d_base = (blockIdx.x * c_num_cols);
        int c_base = c_ptr_base[bid];

        for(int ci = threadIdx.x ; ci < c_ptr_nnz[bid]; ci+= blockDim.x)
        {
            int col = c_idx_dummy[c_base+ci];
            float is_zero = atomicAdd(&dense[d_base + col], c_val[c_base+ci]);
            if(is_zero<0.0001 && is_zero>-0.0001)
            {
                int loc = atomicAdd(&sh_loc, 1);
                c_idx[c_base + loc] = col;
            }
        }
        __syncthreads();
        for(int ci = threadIdx.x ; ci<sh_loc;ci+= blockDim.x)
        {
            int col = c_idx[c_base + ci];
            c_val[c_base + ci] = dense[d_base + col];
            atomicExch(&dense[d_base+col], 0);
        } 
        __syncthreads();
        if(threadIdx.x==0)
            c_ptr_nnz[bid] = sh_loc;
    }
}


void cudaMergeGEMM(cm C)
{
    int num_dense = DENSE_NUM;
    cudaMalloc((void**)&c_idx_dummy, sizeof(int)*cmGetNNZ(C));
    cudaMalloc((void**)&dense, sizeof(float)*cmGetNumCols(C)*num_dense);
    cudaMemset(dense, 0,sizeof(float)*cmGetNumCols(C)*num_dense);
    cudaMemcpy(c_idx_dummy, c_idx, sizeof(int)*cmGetNNZ(C),cudaMemcpyDeviceToDevice);

    mergeGEMM<<<DENSE_NUM,64>>>(
            c_ptr_base, c_ptr_nnz, c_idx, c_val, c_idx_dummy,
            dense);
    cudaDeviceSynchronize();

    int* t_ptr_nnz = new int[cmGetNumRows(C)];
    int* t_ptr_base = new int[cmGetNumRows(C)+1];
    int* t_idx = new int[cmGetNNZ(C)];
    float* t_val = new float[cmGetNNZ(C)];
    cudaMemcpy(t_ptr_base, c_ptr_base, sizeof(int)*cmGetNumRows(C), cudaMemcpyDeviceToHost);
    cudaMemcpy(t_ptr_nnz, c_ptr_nnz, sizeof(int)*cmGetNumRows(C), cudaMemcpyDeviceToHost);

    int nnzC=0;
    for(int i=0; i<cmGetNumRows(C);i++)
    {
        int c_base = t_ptr_base[i];
        int c_nnz = t_ptr_nnz[i];
        nnzC += c_nnz;
        cudaMemcpy(&t_idx[c_base], &c_idx[c_base], sizeof(int)*c_nnz, cudaMemcpyDeviceToHost);
        cudaMemcpy(&t_val[c_base], &c_val[c_base], sizeof(float)*c_nnz, cudaMemcpyDeviceToHost);
    }
    ERROR_CHECK;

    int nnzCf = 0;
    for(int i=0;i< cmGetNumRows(C);i++)
    {
        int c_base = t_ptr_base[i];
        int c_nnz = t_ptr_nnz[i];
        for(int j=0;j<c_nnz;j++)
        {
            if(t_val[c_base+j]>0.0001)
                nnzCf++;
        }
    }
    printf("%d %d\n",nnzC, nnzCf);
}/*}}}*/

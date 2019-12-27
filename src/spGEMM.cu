#include "../lib/coo.h"
#include "../lib/csc.h"
#include "../lib/csr.h"
#include "../lib/wlt.h"
#include "../lib/thread.h"
#include "../lib/spGEMM.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define SHARED 
#define ERROR_CHECK \
{\
	cudaError err = cudaGetLastError(); \
	if ( cudaSuccess != err ) \
	{\
		printf("[%s:%d]CUDA ERROR : %s\n", __FILE__, __LINE__, cudaGetErrorString(err) ); \
		sleep(5); \
		exit(-1); \
	}\
}

//#define MERGE 

#define M16 1//32
#define M8 1//16
#define M4 1//8
#define M2 1//4
#define M1 1//2 


/*
#define M16 32
#define M8 16
#define M4 8
#define M2 4
#define M1 2 
*/
#define MT 64
#define GRID 1024
#define ALPHA 50 
#define BETA 10

__device__ uint get_smid(void){
	uint ret;
	asm("mov.u32 %0, %smid;" : "=r"(ret));
	return ret;
}

__global__ void calcUpp(
        int* csrCidx,int *csrPtr,
        int* cscRidx,int *cscPtr,
        int *rUpp,
        int *wlt,
		int* N, int * part_out, int* part_row){
    int tid = threadIdx.x;
    int bid = blockIdx.x;   //bid = rowIdx
    int bsize = blockDim.x;
	if(bid<N[0]){
		/* CALCULATE UPPER BOUND */
		int base = csrPtr[bid]; 
		int rLen = csrPtr[bid+1]-base;
		int cLen = cscPtr[bid+1]-cscPtr[bid];

		for(int i = tid ; i < rLen; i+=bsize){
			int target = csrCidx[base+i];
			int len = csrPtr[target+1] - csrPtr[target];
			atomicAdd(&rUpp[bid], csrPtr[target+1]-csrPtr[target]);
		}
		/* BUILD WORKLOADTABLE */
		wlt[bid] = rLen*cLen;
        if(rLen*cLen && tid==0) atomicAdd(&part_out[0],1);
        if(rUpp[bid] && tid==0) atomicAdd(&part_row[0],1);
	}
}

__global__ void findDom(int* csrPtr, int* cscPtr,
		int *rUpp,
		int* wlt,
		int *domCC, int *domRC,
		int *domR, int *domN,
        int *dominator_c,
		int *number_of_dominators_in_c,
        int *else_c,
        int *number_of_else_in_c,
        char *c_bool,
        int *N, 
        int* part_out, 
        int* part_row){

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int bsize = blockDim.x;
	
	int index = bsize*bid + tid;
	if(index < N[0]){
		if(1){//(double)wlt[index]>(rUpp[N[0]]/part_out[0] * ALPHA)){
            int cntr = atomicAdd(&domN[0],1);
			int cc = cscPtr[index+1] - cscPtr[index];
			int rc = csrPtr[index+1] - csrPtr[index];
			domR[cntr] = index;
			wlt[index]=0;
			atomicAdd(&domCC[0],cc);
			atomicAdd(&domRC[0],rc);

		}

        if(rUpp[index+1] - rUpp[index] > rUpp[*N]/part_row[0]*5){
            int temp = atomicAdd(&number_of_dominators_in_c[0], 1);
            dominator_c[temp] = index;
 //           c_bool[index] = 1;
        }
        else{
            int temp = atomicAdd(&number_of_else_in_c[0],1);
            else_c[temp] = index;
        }
	}
}

__global__ void find_thread_count_approx(
        int* a_ptr, int* b_ptr,int* thread_bin, int *N)
{
    int global_tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(global_tid<N[0]){
        int len = b_ptr[global_tid+1] - b_ptr[global_tid];
        if(0<len && len <=2)                 atomicAdd(&thread_bin[0],1);
        else if(2<len && len <=4)            atomicAdd(&thread_bin[1],1);
        else if(4<len && len <=8)           atomicAdd(&thread_bin[2],1);
        else if(8<len && len <=16)          atomicAdd(&thread_bin[3],1);
        else if(16<len && len <=32)          atomicAdd(&thread_bin[4],1);
        else if(32<len && len <=64)         atomicAdd(&thread_bin[5],1);
        else if(64<len && len <= 128)       atomicAdd(&thread_bin[6],1);
        else if(128<len)                 atomicAdd(&thread_bin[7],1);

    }
}

__global__ void fill_thread_bin_approx(
        int* a_ptr, int* b_ptr,
        int* thread_bin, int* thread_counter,
        int* index_bin, int* N)
{
    int global_tid = threadIdx.x + blockIdx.x*blockDim.x;
    if(global_tid < N[0]){
        int len = b_ptr[global_tid+1] - b_ptr[global_tid];
        if(0<len && len <=2){
            int a = atomicAdd(&thread_counter[0], 1);
            index_bin[thread_bin[0] + a] = global_tid;
        }
        else if(2<len && len <=4){
            int a = atomicAdd(&thread_counter[1], 1);
            index_bin[thread_bin[1] + a] = global_tid;
        }
        else if(4<len && len <=8){
            int a = atomicAdd(&thread_counter[2], 1);
            index_bin[thread_bin[2] + a] = global_tid;
        }
        else if(8<len && len <=16){
            int a = atomicAdd(&thread_counter[3], 1);
            index_bin[thread_bin[3] + a]=global_tid;
        }
        else if(16<len && len <=32){
            int a = atomicAdd(&thread_counter[4], 1);
            index_bin[thread_bin[4] + a] = global_tid;
        }
        else if(32<len && len <=64){
            int a = atomicAdd(&thread_counter[5], 1);
            index_bin[thread_bin[5] + a] = global_tid;
        }
        else if(64<len && len <=128){
            int a = atomicAdd(&thread_counter[6], 1);
            index_bin[thread_bin[6] + a] =global_tid;
        }
        else if(128< len ){
            int a = atomicAdd(&thread_counter[7], 1);
            index_bin[thread_bin[7] + a] =global_tid;
        }
    }
}
__global__ void calcInterBase(
        int* idx_bin,
        int* csrCidx, int* csrPtr, float* csrVal,       // MATRIX B(CSR)
		int* cscRidx, int* cscPtr, float* cscVal,       // MATRIX A(CSC)
		int* c_idx, int* c_ptr, float* c_val,
		int *RUPP,
        int *WLS,
        int *RBOX)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
    int bsize = blockDim.x;
    int idx = idx_bin[bid];

    if(WLS[idx]==0){asm("exit;");}

        int csrBase = csrPtr[idx];
        int cscBase = cscPtr[idx];
        int rLen = csrPtr[idx+1] - csrBase;
        int cLen = cscPtr[idx+1] - cscBase;

        __shared__ int resBase;
        __syncthreads();
        for( int ci = 0  ; ci < cLen; ci ++)
        {
            int rIdx = cscRidx[ cscBase + ci ];
            int rowBase = RUPP[rIdx];

            if(tid==0) 
                resBase = atomicAdd(&RBOX[rIdx], rLen);
            
            __syncthreads();
            for(int ri= tid ;ri < rLen ; ri+=bsize)
            {
                int cIdx = csrCidx[ csrBase + ri ];

                c_idx[rowBase+resBase+ri] = cIdx;
                c_val[rowBase+resBase+ri] = cscVal[ cscBase + ci ]*csrVal[ csrBase + ri ];
            }
        }

}


__global__ void calcInterNoMerge(
        int* idx_bin,
        int* csrCidx, int* csrPtr, float* csrVal,       // MATRIX B(CSR)
		int* cscRidx, int* cscPtr, float* cscVal,       // MATRIX A(CSC)
		int* c_idx, int* c_ptr, float* c_val,
		int *RUPP,
        int *WLS,
        int *RBOX)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
    int bsize = blockDim.x;

    int newTid=tid;
    int newBid=0;
    int idx = idx_bin[bid];

    if(WLS[idx]==0){asm("exit;");}
    else{

        int csrBase = csrPtr[idx];
        int cscBase = cscPtr[idx];
        int rLen = csrPtr[idx+1] - csrBase;
        int cLen = cscPtr[idx+1] - cscBase;

        __shared__ int resBase[16];
        __syncthreads();
        for( int ci = 0  ; ci < cLen; ci ++)
        {
            int rIdx = cscRidx[ cscBase + ci ];
            int rowBase = RUPP[rIdx];

            if(newTid==0) 
                resBase[newBid] = atomicAdd(&RBOX[rIdx], rLen);
            
            __syncthreads();
            for(int ri= newTid ;ri < rLen ; ri+=bsize)
            {
                int cIdx = csrCidx[ csrBase + ri ];

                c_idx[rowBase+resBase[newBid]+ri] = cIdx;
                c_val[rowBase+resBase[newBid]+ri] = cscVal[ cscBase + ci ]*csrVal[ csrBase + ri ];
            }
        }
        WLS[idx] = 0; // fuck
    }
}

__global__ void calcInterMerge1(
        int* idx_bin,
        int* csrCidx, int* csrPtr, float* csrVal,       // MATRIX B(CSR)
		int* cscRidx, int* cscPtr, float* cscVal,       // MATRIX A(CSC)
		int* c_idx, int* c_ptr, float* c_val,
		int *RUPP,
        int *WLS,
        int *RBOX)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
    int bsize = blockDim.x;

    int newTid=tid;
    int newBid=0;

    bid *=M1;
    newBid = tid / (bsize/M1);
    newTid = tid - (bsize/M1)*newBid;

    int idx = idx_bin[bid+newBid];

    if(WLS[idx]==0){asm("exit;");}
    else{

        int csrBase = csrPtr[idx];
        int cscBase = cscPtr[idx];
        int rLen = csrPtr[idx+1] - csrBase;
        int cLen = cscPtr[idx+1] - cscBase;

        __shared__ int resBase[32];
        __syncthreads();
        for( int ci = 0  ; ci < cLen; ci ++)
        {
            int rIdx = cscRidx[ cscBase + ci ];
            int rowBase = RUPP[rIdx];

            if(newTid==0) 
                resBase[newBid] = atomicAdd(&RBOX[rIdx], rLen);
            
            __syncthreads();
            for(int ri= newTid ;ri < rLen ; ri+=bsize/M1)
            {
                int cIdx = csrCidx[ csrBase + ri ];

                c_idx[rowBase+resBase[newBid]+ri] = cIdx;
                c_val[rowBase+resBase[newBid]+ri] = cscVal[ cscBase + ci ]*csrVal[ csrBase + ri ];
            }
        }
        WLS[idx] = 0; // fuck
    }
}



__global__ void calcInterMerge2(
        int* idx_bin,
        int* csrCidx, int* csrPtr, float* csrVal,       // MATRIX B(CSR)
		int* cscRidx, int* cscPtr, float* cscVal,       // MATRIX A(CSC)
		int* c_idx, int* c_ptr, float* c_val,
		int *RUPP,
        int *WLS,
        int *RBOX)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
    int bsize = blockDim.x;

    int newTid=tid;
    int newBid=0;

    bid *=M2;
    newBid = tid / (bsize/M2);
    newTid = tid - (bsize/M2)*newBid;

    int idx = idx_bin[bid+newBid];

    if(WLS[idx]==0){asm("exit;");}
    else{

        int csrBase = csrPtr[idx];
        int cscBase = cscPtr[idx];
        int rLen = csrPtr[idx+1] - csrBase;
        int cLen = cscPtr[idx+1] - cscBase;

        __shared__ int resBase[32];
        __syncthreads();
        for( int ci = 0  ; ci < cLen; ci ++)
        {
            int rIdx = cscRidx[ cscBase + ci ];
            int rowBase = RUPP[rIdx];

            if(newTid==0) 
                resBase[newBid] = atomicAdd(&RBOX[rIdx], rLen);
            
            __syncthreads();
            for(int ri= newTid ;ri < rLen ; ri+=bsize/M2)
            {
                int cIdx = csrCidx[ csrBase + ri ];

                c_idx[rowBase+resBase[newBid]+ri] = cIdx;
                c_val[rowBase+resBase[newBid]+ri] = cscVal[ cscBase + ci ]*csrVal[ csrBase + ri ];
            }
        }
        WLS[idx] = 0; // fuck
    }
}

__global__ void calcInterMerge4(
        int* idx_bin,
        int* csrCidx, int* csrPtr, float* csrVal,       // MATRIX B(CSR)
		int* cscRidx, int* cscPtr, float* cscVal,       // MATRIX A(CSC)
		int* c_idx, int* c_ptr, float* c_val,
		int *RUPP,
        int *WLS,
        int *RBOX)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
    int bsize = blockDim.x;
    
    int newTid = tid;
    int newBid = 0;

    bid*= M4;
    newBid = tid / (bsize/M4);
    newTid = tid - (bsize/M4)*newBid;
    
    int idx = idx_bin[bid + newBid];


    if(WLS[idx]==0){asm("exit;");}
    else{

        int csrBase = csrPtr[idx];
        int cscBase = cscPtr[idx];
        int rLen = csrPtr[idx+1] - csrBase;
        int cLen = cscPtr[idx+1] - cscBase;

        __shared__ int resBase[32];
        __syncthreads();
        for( int ci = 0  ; ci < cLen; ci ++)
        {
            int rIdx = cscRidx[ cscBase + ci ];
            int rowBase = RUPP[rIdx];

            if(newTid==0) 
                resBase[newBid] = atomicAdd(&RBOX[rIdx], rLen);
            
            __syncthreads();
            for(int ri= newTid ;ri < rLen ; ri+=bsize/M4)
            {
                int cIdx = csrCidx[ csrBase + ri ];

                c_idx[rowBase+resBase[newBid]+ri] = cIdx;
                c_val[rowBase+resBase[newBid]+ri] = cscVal[ cscBase + ci ]*csrVal[ csrBase + ri ];
            }
        }
        WLS[idx] = 0; // fuck
    }
}

__global__ void calcInterMerge8(
        int* idx_bin,
        int* csrCidx, int* csrPtr, float* csrVal,       // MATRIX B(CSR)
		int* cscRidx, int* cscPtr, float* cscVal,       // MATRIX A(CSC)
		int* c_idx, int* c_ptr, float* c_val,
		int *RUPP,
        int *WLS,
        int *RBOX)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
    int bsize = blockDim.x;

    int newBid = 0;
    int newTid = tid;
    
    bid*= M8;
    newBid = tid / (bsize/M8);
    newTid = tid - (bsize/M8)*newBid;

    int idx = idx_bin[bid + newBid];

    if(WLS[idx]==0){asm("exit;");}
    else{

        int csrBase = csrPtr[idx];
        int cscBase = cscPtr[idx];
        int rLen = csrPtr[idx+1] - csrBase;
        int cLen = cscPtr[idx+1] - cscBase;

        __shared__ int resBase[32];
        __syncthreads();
        for( int ci = 0  ; ci < cLen; ci ++)
        {
            int rIdx = cscRidx[ cscBase + ci ];
            int rowBase = RUPP[rIdx];

            if(newTid==0) 
                resBase[newBid] = atomicAdd(&RBOX[rIdx], rLen);
            
            __syncthreads();
            for(int ri= newTid ;ri < rLen ; ri+=bsize/M8)
            {
                int cIdx = csrCidx[ csrBase + ri ];

                c_idx[rowBase+resBase[newBid]+ri] = cIdx;
                c_val[rowBase+resBase[newBid]+ri] = cscVal[ cscBase + ci ]*csrVal[ csrBase + ri ];
            }
        }
       WLS[idx] = 0; // fuck
    }
}



__global__ void calcInterMerge16(
        int* idx_bin,
        int* csrCidx, int* csrPtr, float* csrVal,       // MATRIX B(CSR)
		int* cscRidx, int* cscPtr, float* cscVal,       // MATRIX A(CSC)
		int* c_idx, int* c_ptr, float* c_val,
		int *RUPP,
        int *WLS,
        int *RBOX)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
    int bsize = blockDim.x;

    int newBid = 0;
    int newTid = tid;
    
    bid*= M16;
    newBid = tid / (bsize/M16);
    newTid = tid - (bsize/M16)*newBid;

    int idx = idx_bin[bid + newBid];

    if(WLS[idx]==0){asm("exit;");}
    else{

        int csrBase = csrPtr[idx];
        int cscBase = cscPtr[idx];
        int rLen = csrPtr[idx+1] - csrBase;
        int cLen = cscPtr[idx+1] - cscBase;

        __shared__ int resBase[32];
        __syncthreads();
        for( int ci = 0  ; ci < cLen; ci ++)
        {
            int rIdx = cscRidx[ cscBase + ci ];
            int rowBase = RUPP[rIdx];

            if(newTid==0) 
                resBase[newBid] = atomicAdd(&RBOX[rIdx], rLen);
            
            __syncthreads();
            for(int ri= newTid ;ri < rLen ; ri+=bsize/M16)
            {
                int cIdx = csrCidx[ csrBase + ri ];

                c_idx[rowBase+resBase[newBid]+ri] = cIdx;
                c_val[rowBase+resBase[newBid]+ri] = cscVal[ cscBase + ci ]*csrVal[ csrBase + ri ];
            }
        }
       WLS[idx] = 0; // fuck
    }
}

__global__ void calcInterDom(
		int* csrCidx, int* csrPtr, float* csrVal,       // MATRIX B(CSR)
		int* cscRidx, int* cscPtr, float* cscVal,       // MATRIX A(CSC)
		int* c_idx, int* c_ptr, float* c_val,
        // cooData *interC,                                // INTERMEDIATE C
		int *RUPP,
        PP *P,
        int *RBOX)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
    int bsize = blockDim.x;
    
    int rid = P[bid].r;
    int cid = P[bid].c;
    
 	
    int csrBase = csrPtr[rid];                   
    int cscBase = cscPtr[cid];           
    int rLen = csrPtr[rid+1] - csrBase;
    int cLen = cscPtr[cid+1] - cscBase;
   __shared__ int resBase;
    resBase = 0;

    for( int ci=0 ; ci < cLen; ci ++)
    {
        __syncthreads();

        int rIdx = cscRidx[ cscBase + ci ];
        int rowBase = RUPP[rIdx];
        if(tid==0) resBase = atomicAdd(&RBOX[rIdx], rLen);
        __syncthreads();

        for(int ri = tid; ri < rLen ; ri+=bsize )
        {
            int cIdx = csrCidx[ csrBase + ri ];

            //interC[rowBase + resBase + ri].cidx = cIdx;
            //interC[rowBase + resBase + ri].ridx = rIdx;    
            //interC[rowBase + resBase + ri].val =  cscVal[ cscBase + ci ] * csrVal[ csrBase + ri ];
            c_idx[rowBase+resBase+ri] = cIdx;
            c_val[rowBase+resBase+ri] = cscVal[ cscBase + ci ] * csrVal[ csrBase + ri ];
        }
    }

    
 }



__global__ void merge_limitting(
                //const cooData *interC,    // INTERMEDIATE C
                int *RUPP,
                int *RBOX,                          
                float* DROW,                        // DENSE ROW   
                //int* MIDX, float* MVAL, int* MPTR,  // MERGED C
                int* c_jdx, float* c_val, int* c_ptr,
                int *c_idx,
                int* dominator_c,
                int* number_of_dominators_in_c,
                int *N)         
{   
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bsize = blockDim.x;
 
    __shared__ int ip;
    __shared__ double shm[768*4];
    shm[0] = 0;
    
    for(int RR = 0 ; RR<(*N/GRID)+1; RR++){
        if(bid+(RR)*GRID>=*number_of_dominators_in_c) return;

        int ii = dominator_c[bid+RR*GRID];    
        int rBase = RUPP[ii];
        int rLen  = RUPP[ii+1]-RUPP[ii];


        int dBase = bid*(*N);
        
        __syncthreads();
        ip = 0;
        __syncthreads();
        for(int i=tid; i<rLen;i+=bsize){
            int index = c_jdx[rBase+i];
            float boolflag = atomicExch(&DROW[dBase+index], DROW[dBase+index] + c_val[rBase+i]);
            if(boolflag<0.0001 && boolflag>-0.0001){
                int ip_local = atomicAdd(&ip,1);
                c_idx[rBase+ip_local] = index;//c_idx[rBase+i];
                //atomicExch(&c_idx[rBase+ip_local],index);
            }
        }
        __syncthreads();
        for(int i=tid;i<ip;i+=bsize){
            int v = c_idx[rBase + i];
            c_val[rBase + i] = DROW[dBase+v];
            atomicExch(&DROW[dBase+v], 0);
        }
        __syncthreads();

        if(tid==0) c_ptr[ii] = ip;
    }
}

__global__ void merge(
                //const cooData *interC,    // INTERMEDIATE C
                int *RUPP,
                int *RBOX,                          
                float* DROW,                        // DENSE ROW   
                //int* MIDX, float* MVAL, int* MPTR,  // MERGED C
                int* c_jdx, float* c_val, int* c_ptr,
                int *c_idx,
                int* else_c,
                int* number_of_else_in_c,
                int *N)         
{   
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bsize = blockDim.x;
//    __shared__ double shm[3*1024] ;
//    shm[0] = 0;
    __shared__ int ip;
    
    for(int RR = 0 ; RR<(*N/GRID)+1; RR++){
        if(bid+(RR)*GRID>=*number_of_else_in_c) return;
        int ii = else_c[bid+RR*GRID];
        
        int rBase = RUPP[ii];
        int rLen  = RUPP[ii+1]-RUPP[ii];


        int dBase = bid*(*N);
        
        __syncthreads();
        ip = 0;
        __syncthreads();
        for(int i=tid; i<rLen;i+=bsize){
            int index = c_jdx[rBase+i];
            float boolflag = atomicExch(&DROW[dBase+index], DROW[dBase+index] + c_val[rBase+i]);
            if(boolflag<0.0001 && boolflag>-0.0001){
                int ip_local = atomicAdd(&ip,1);
                c_idx[rBase+ip_local] = index;//c_idx[rBase+i];
                //atomicExch(&c_idx[rBase+ip_local],index);
            }
        }
        __syncthreads();
        for(int i=tid;i<ip;i+=bsize){
            int v = c_idx[rBase + i];
            c_val[rBase + i] = DROW[dBase+v];
            atomicExch(&DROW[dBase+v], 0);
        }
        __syncthreads();

        if(tid==0) c_ptr[ii] = ip;
    }
}

int *thread_counter_d, *thread_counter_h;
int *thread_bin_d, *thread_bin_h;
int *idx_bin_d, *idx_bin_h;
int *participants_d;
void cudaPath_find_thread_count_approx(int n)
{
    int block_size = 32;
    int block_num  = n/32+1;
    float time;
    cudaMalloc((void**) &(thread_bin_d),     sizeof(int) *10);    
    cudaMalloc((void**) &(thread_counter_d), sizeof(int) *10);    
    cudaMemset(thread_bin_d, 0,     sizeof(int)*10);
    cudaMemset(thread_counter_d, 0, sizeof(int)*10);
   
    cudaEvent_t ev_count_thread_s, ev_count_thread_e;
    cudaEventCreate(&ev_count_thread_s);
    cudaEventCreate(&ev_count_thread_e);
    cudaEventRecord(ev_count_thread_s, 0);



    find_thread_count_approx<<<block_num,block_size>>>
        (CSC_PTR_DEV,
         CSR_PTR_DEV,
         thread_bin_d,N_DEV);
    thread_bin_h = (int*)malloc(sizeof(int)*10);
    thread_counter_h = (int*)malloc(sizeof(int)*10);

    cudaMemcpy((void*)&thread_bin_h[1], (const void*)thread_bin_d, sizeof(int)*8,cudaMemcpyDeviceToHost); ERROR_CHECK; 

    int total = 0;
    thread_bin_h[0] = 0;
    thread_bin_h[8] = 0;
    for(int i=0;i<8;i++){
        thread_bin_h[i+1] += thread_bin_h[i];
    }
    total = thread_bin_h[8];
    cudaMemcpy((void*)thread_bin_d, (const void*)thread_bin_h, sizeof(int)*8,cudaMemcpyHostToDevice); ERROR_CHECK;


    cudaMalloc((void**) &(idx_bin_d), sizeof(int)*thread_bin_h[8]);    ERROR_CHECK;
    idx_bin_h = (int*)malloc(sizeof(int)*thread_bin_h[8]);
    fill_thread_bin_approx<<<block_num, block_size>>>
        (CSC_PTR_DEV, CSR_PTR_DEV,
         thread_bin_d, thread_counter_d,
         idx_bin_d,N_DEV);
    ERROR_CHECK;

	cudaDeviceSynchronize();
    cudaEventRecord(ev_count_thread_e, 0);
	cudaEventSynchronize(ev_count_thread_e);
    cudaEventElapsedTime(&t_bin,ev_count_thread_s, ev_count_thread_e);
    
    
    cudaMemcpy((void*)idx_bin_h, (const void*)idx_bin_d, sizeof(int)*thread_bin_h[8],cudaMemcpyDeviceToHost); ERROR_CHECK;
    cudaMemcpy((void*)thread_counter_h, (const void*)thread_counter_d, sizeof(int)*10,cudaMemcpyDeviceToHost); ERROR_CHECK;

    int counter = 0;
 /*   for(int i=0;i<8;i++){
        printf("%d ",thread_counter_h[i]);
        int a = thread_bin_h[i+1] - thread_bin_h[i];
        for(int j =0;j<a;j++){
            printf("%d %d\n",counter++,idx_bin_h[j]);
        }
        
    }
    printf("\n");*/
    
    

}
int* participants_row_d;
void cudaPass_(int n, int e){	
    int BLOCK_SIZE=32;
    int BLOCK_NUM = n;
	dim3 block(BLOCK_SIZE);
    dim3 grid(BLOCK_NUM);
	
    /* CUDAMALLOC & CUDAMEMCPY B(CSR) */
    cudaMalloc((void**) &(CSR_CIDX_DEV), sizeof(int)  *e);    
    cudaMalloc((void**) &(CSR_VAL_DEV),  sizeof(float)*e);    
    cudaMalloc((void**) &(CSR_PTR_DEV),  sizeof(int)  *(2*n+1));    
	cudaMemcpy((void*) (CSR_CIDX_DEV),	(const void*)(CSR_HOST.cidx),	sizeof(int)    *e,	    cudaMemcpyHostToDevice);
  	cudaMemcpy((void*) (CSR_VAL_DEV),	(const void*)(CSR_HOST.val),	sizeof(float)  *e,	    cudaMemcpyHostToDevice);
    cudaMemcpy((void*) (CSR_PTR_DEV),	(const void*)(CSR_HOST.header),	sizeof(int)    *(n+1),	cudaMemcpyHostToDevice);
    
    /* CUDAMALLOC & CUDAMEMCPY A(CSC) */
    cudaMalloc((void**) &(CSC_RIDX_DEV), sizeof(int)  *e);    
    cudaMalloc((void**) &(CSC_VAL_DEV),  sizeof(float)*e);    
    cudaMalloc((void**) &(CSC_PTR_DEV),  sizeof(int)  *(2*n+1));    
    cudaMemcpy((void*) (CSC_RIDX_DEV),	(const void*)(CSC_HOST.ridx),	sizeof(int)    *e,	    cudaMemcpyHostToDevice);
    cudaMemcpy((void*) (CSC_VAL_DEV),	(const void*)(CSC_HOST.val),	sizeof(float)  *e,	    cudaMemcpyHostToDevice);
  	cudaMemcpy((void*) (CSC_PTR_DEV),	(const void*)(CSC_HOST.header), sizeof(int)    *(n+1),	cudaMemcpyHostToDevice);
	
    cudaMalloc((void**) &(RUPP_DEV), sizeof(int)*(n+1));
    cudaMalloc((void**) &(WLS_DEV),  sizeof(int)*(n+1));
    cudaMemset(RUPP_DEV, 0, sizeof(int)*(n+1));

	cudaMalloc((void**) &(N_DEV),sizeof(int));
	cudaMemcpy((void*)N_DEV, (const void*)&n ,sizeof(int),cudaMemcpyHostToDevice);
    cudaMalloc((void**) &(participants_d),  sizeof(int));
    cudaMalloc((void**) &(participants_row_d),  sizeof(int));//row

    cudaMemset(RUPP_DEV, 0, sizeof(int));

	cudaEvent_t ev_pre_s, ev_pre_e;
    cudaEventCreate(&ev_pre_s);
    cudaEventCreate(&ev_pre_e);
    cudaEventRecord(ev_pre_s, 0);
    calcUpp<<<grid, block>>>(
        CSR_CIDX_DEV, CSR_PTR_DEV,
        CSC_RIDX_DEV, CSC_PTR_DEV,
        RUPP_DEV,
        WLS_DEV,
		N_DEV,
        participants_d, participants_row_d);
	
	//ERROR_CHECK;
    cudaDeviceSynchronize();
    cudaEventRecord(ev_pre_e, 0);
	cudaEventSynchronize(ev_pre_e);
    cudaEventElapsedTime(&t_pre,ev_pre_s, ev_pre_e);
    
	RUPP_HOST = (int*)malloc(sizeof(int)*(n+1));
	cudaMallocHost((void**)&(WLS_HOST),sizeof(int)*(n+1));
    cudaMemcpy((void*)&RUPP_HOST[1], (const void*)RUPP_DEV, sizeof(int)*n, cudaMemcpyDeviceToHost);
}

int* number_of_dominators_in_c;
int* number_of_dominators_in_c_h;
int* dominator_c;
int* dominator_c_h;
int* else_c;
int *number_of_else_in_c;
char* c_bool;
void cudaPass_F(int n){
	int* domRC_dev;
	int* domCC_dev;
	int* domN_dev;
	int* domR_dev;

	cudaMemcpy((void*)RUPP_DEV,(const void*)RUPP_HOST, sizeof(int)*(n+1), cudaMemcpyHostToDevice); ERROR_CHECK;

	cudaMalloc((void**)&(domR_dev), sizeof(int)*100000);ERROR_CHECK;
	cudaMalloc((void**)&(domRC_dev), sizeof(int));ERROR_CHECK;
	cudaMalloc((void**)&(domCC_dev), sizeof(int));ERROR_CHECK;
    cudaMalloc((void**)&(dominator_c),sizeof(int)*n);
    cudaMalloc((void**)&(else_c),sizeof(int)*n);

	cudaMalloc((void**)&(domN_dev), sizeof(int)) ;ERROR_CHECK;
    cudaMalloc((void**)&(number_of_dominators_in_c), sizeof(int)); ERROR_CHECK;
    cudaMalloc((void**)&(number_of_else_in_c), sizeof(int)); ERROR_CHECK;

    cudaMalloc((void**)&(c_bool), sizeof(char)*n); ERROR_CHECK;

    cudaMemset(domRC_dev, 0, sizeof(int));
    cudaMemset(domCC_dev, 0, sizeof(int));
    cudaMemset(domN_dev, 0, sizeof(int));
    cudaMemset(number_of_dominators_in_c, 0, sizeof(int));
    cudaMemset(number_of_else_in_c, 0, sizeof(int));

    cudaMemset(c_bool, 0, sizeof(char)*n);

	//printf("%d\n",n);
	int grid = n/256+1;
	int block = 256;

	findDom<<<grid, block>>>(
			CSR_PTR_DEV, CSC_PTR_DEV,
			RUPP_DEV,
			WLS_DEV,
			domCC_dev, domRC_dev,
			domR_dev, domN_dev,
            dominator_c,
			number_of_dominators_in_c,
            else_c,
            number_of_else_in_c,
            c_bool,
            N_DEV,
            participants_d,participants_row_d); 
	ERROR_CHECK;

    cudaDeviceSynchronize();
	cudaEvent_t ev_fd_s, ev_fd_e;
    cudaEventCreate(&ev_fd_s);
    cudaEventCreate(&ev_fd_e);
    cudaEventRecord(ev_fd_s, 0);

    cudaMemcpy((void*)domR,  (const void*)domR_dev,  sizeof(int)*100000,cudaMemcpyDeviceToHost); ERROR_CHECK;
	cudaMemcpy((void*)domRC, (const void*)domRC_dev, sizeof(int),cudaMemcpyDeviceToHost); ERROR_CHECK;
	cudaMemcpy((void*)domCC, (const void*)domCC_dev, sizeof(int),cudaMemcpyDeviceToHost); ERROR_CHECK;
	cudaMemcpy((void*)domN,  (const void*)domN_dev,  sizeof(int),cudaMemcpyDeviceToHost); ERROR_CHECK;
    

    dominator_c_h = (int*)malloc(sizeof(int)*n);
    number_of_dominators_in_c_h = (int*)malloc(sizeof(int));

    cudaMemcpy((void*)dominator_c_h,  (const void*)dominator_c,  sizeof(int)*n, cudaMemcpyDeviceToHost); ERROR_CHECK;
    cudaMemcpy((void*)number_of_dominators_in_c_h, number_of_dominators_in_c, sizeof(int),cudaMemcpyDeviceToHost);
    if(number_of_dominators_in_c_h[0]!=0) printf("!\n");
	cudaEventRecord(ev_fd_e, 0);
	cudaEventSynchronize(ev_fd_e);
    cudaEventElapsedTime(&t_fd,ev_fd_s, ev_fd_e);

}

void cudaPassB(int n){	
    int BLOCK_SIZE = 256;

	dim3 block(BLOCK_SIZE);
    dim3 grid(thread_counter_h[0]);
    cudaMalloc((void**) &(c_jdx_d),  sizeof(int)*RUPP_HOST[n]);
    cudaMalloc((void**) &(c_idx_d),  sizeof(int)*RUPP_HOST[n]); 
    cudaMalloc((void**) &(c_val_d),  sizeof(float)*RUPP_HOST[n]); 
    cudaMalloc((void**) &(c_ptr_d),  sizeof(int)*(n+1)); 
    
    cudaMalloc((void**) &(RBOX_DEV), sizeof(int)*n);
    cudaMemset(RBOX_DEV, 0, sizeof(int)*n);

	
    cudaEvent_t ev_spgemm_s, ev_spgemm_e;
    cudaEventCreate(&ev_spgemm_s);
    cudaEventCreate(&ev_spgemm_e);
    cudaEventRecord(ev_spgemm_s, 0);
   /* 
     calcInterBase<<<grid,block>>>(
            &idx_bin_d[0],
                CSR_CIDX_DEV, CSR_PTR_DEV, CSR_VAL_DEV,     // MATRIX B(CSR)
                CSC_RIDX_DEV, CSC_PTR_DEV, CSC_VAL_DEV,     // MATRIX A(CSC)
                c_jdx_d, c_ptr_d, c_val_d,                  // MATRIX C(CSR)
                RUPP_DEV,                                   // UPPER BOUND           
                WLS_DEV,                                    // WLS_DEV
                RBOX_DEV);
        cudaDeviceSynchronize();


       cudaDeviceSynchronize();
       cudaEventRecord(ev_spgemm_e, 0);
       cudaEventSynchronize(ev_spgemm_e);
    cudaEventElapsedTime(&t_spgemm_l,ev_spgemm_s, ev_spgemm_e);
    return;
*/
    int off;
    off = thread_counter_h[0];
    //printf("%d\n",thread_counter_h[0]);
    if(off){
        dim3 block_num0(off/M16+1); // mergefactor  : 8
        //dim3 block_num0(off); // mergefactor  : 8

        calcInterMerge16<<<block_num0,block>>>(//block>>>(
        //calcInterNoMerge<<<block_num0,block>>>(

            &idx_bin_d[0],
                CSR_CIDX_DEV, CSR_PTR_DEV, CSR_VAL_DEV,     // MATRIX B(CSR)
                CSC_RIDX_DEV, CSC_PTR_DEV, CSC_VAL_DEV,     // MATRIX A(CSC)
                c_jdx_d, c_ptr_d, c_val_d,                  // MATRIX C(CSR)
                RUPP_DEV,                                   // UPPER BOUND           
                WLS_DEV,                                    // WLS_DEV
                RBOX_DEV);
        cudaDeviceSynchronize();
        ERROR_CHECK;
    }
    off = thread_counter_h[1]; 
    //printf("%d\n",off);
    if(off){
        //printf("2\n");
        dim3 block_num1(off/M8+1);  // merge factor : 4
        //dim3 block_num1(off);  // merge factor : 4

        calcInterMerge8<<<block_num1,block>>>(
        //calcInterNoMerge<<<block_num1,block>>>(
                &idx_bin_d[thread_bin_h[1]],
                CSR_CIDX_DEV, CSR_PTR_DEV, CSR_VAL_DEV,     // MATRIX B(CSR)
                CSC_RIDX_DEV, CSC_PTR_DEV, CSC_VAL_DEV,     // MATRIX A(CSC)
                c_jdx_d, c_ptr_d, c_val_d,                  // MATRIX C(CSR)
                RUPP_DEV,                                   // UPPER BOUND           
                WLS_DEV,                                    // WLS_DEV
                RBOX_DEV);
        cudaDeviceSynchronize();
        ERROR_CHECK;
    }
    off = thread_counter_h[2]; 

    //off = thread_bin_h[3] - thread_bin_h[2]; 
    //printf("%d\n",off);
    if(off){
        //printf("3\n");
        dim3 block_num2(off/M4+1);

        calcInterMerge4<<<block_num2,block>>>(
                &idx_bin_d[thread_bin_h[2]],
                CSR_CIDX_DEV, CSR_PTR_DEV, CSR_VAL_DEV,     // MATRIX B(CSR)
                CSC_RIDX_DEV, CSC_PTR_DEV, CSC_VAL_DEV,     // MATRIX A(CSC)
                c_jdx_d, c_ptr_d, c_val_d,                  // MATRIX C(CSR)
                RUPP_DEV,                                   // UPPER BOUND           
                WLS_DEV,                                    // WLS_DEV
                RBOX_DEV);
        cudaDeviceSynchronize();
        ERROR_CHECK;
    }
    //off = thread_bin_h[4] - thread_bin_h[3]; i
    off = thread_counter_h[3]; 

   // printf("%d\n",off);
    if(off){
        //printf("4\n");
        dim3 block_num3(off/M2+1);

        calcInterMerge2<<<block_num3,block>>>(
                &idx_bin_d[thread_bin_h[3]],
                CSR_CIDX_DEV, CSR_PTR_DEV, CSR_VAL_DEV,     // MATRIX B(CSR)
                CSC_RIDX_DEV, CSC_PTR_DEV, CSC_VAL_DEV,     // MATRIX A(CSC)
                c_jdx_d, c_ptr_d, c_val_d,                  // MATRIX C(CSR)
                RUPP_DEV,                                   // UPPER BOUND           
                WLS_DEV,                                    // WLS_DEV
                RBOX_DEV);
        cudaDeviceSynchronize();
        ERROR_CHECK;
    }

    //off = thread_bin_h[5] - thread_bin_h[4];
    off = thread_counter_h[4]; 

    //printf("%d\n",off);
    if(off){
        //printf("5\n");
        dim3 block_num4(off/M1+1);

        calcInterMerge1<<<block_num4,block>>>(
                &idx_bin_d[thread_bin_h[4]],
                CSR_CIDX_DEV, CSR_PTR_DEV, CSR_VAL_DEV,     // MATRIX B(CSR)
                CSC_RIDX_DEV, CSC_PTR_DEV, CSC_VAL_DEV,     // MATRIX A(CSC)
                c_jdx_d, c_ptr_d, c_val_d,                  // MATRIX C(CSR)
                RUPP_DEV,                                   // UPPER BOUND           
                WLS_DEV,                                    // WLS_DEV
                RBOX_DEV);
        cudaDeviceSynchronize();
        ERROR_CHECK;
    }
    //off = thread_bin_h[6] - thread_bin_h[5];
    off = thread_counter_h[5]; 

    //printf("%d\n",off);
    if(off){
        //printf("6\n");
        dim3 block_size64(64);
        dim3 block_num5(off);

        calcInterNoMerge<<<block_num5,block_size64>>>(
                &idx_bin_d[thread_bin_h[5]],
                CSR_CIDX_DEV, CSR_PTR_DEV, CSR_VAL_DEV,     // MATRIX B(CSR)
                CSC_RIDX_DEV, CSC_PTR_DEV, CSC_VAL_DEV,     // MATRIX A(CSC)
                c_jdx_d, c_ptr_d, c_val_d,                  // MATRIX C(CSR)
                RUPP_DEV,                                   // UPPER BOUND           
                WLS_DEV,                                    // WLS_DEV
                RBOX_DEV);
        cudaDeviceSynchronize();
        ERROR_CHECK;
    }
    //off = thread_bin_h[7] - thread_bin_h[6];
    off = thread_counter_h[6]; 

    //printf("%d\n",off);
    if(off){
        dim3 block_size128(128);
        dim3 block_num6(off);

        calcInterNoMerge<<<block_num6,block_size128>>>(
                &idx_bin_d[thread_bin_h[6]],
                CSR_CIDX_DEV, CSR_PTR_DEV, CSR_VAL_DEV,     // MATRIX B(CSR)
                CSC_RIDX_DEV, CSC_PTR_DEV, CSC_VAL_DEV,     // MATRIX A(CSC)
                c_jdx_d, c_ptr_d, c_val_d,                  // MATRIX C(CSR)
                RUPP_DEV,                                   // UPPER BOUND           
                WLS_DEV,                                    // WLS_DEV
                RBOX_DEV);
        cudaDeviceSynchronize();
        ERROR_CHECK;
    }
    //off = thread_bin_h[8] - thread_bin_h[7];
    off = thread_counter_h[7]; 

    //printf("%d\n",off);
    if(off){
        dim3 block_size256(128);
        dim3 block_num7(off);

        calcInterNoMerge<<<block_num7,block_size256>>>(
                &idx_bin_d[thread_bin_h[7]],
                CSR_CIDX_DEV, CSR_PTR_DEV, CSR_VAL_DEV,     // MATRIX B(CSR)
                CSC_RIDX_DEV, CSC_PTR_DEV, CSC_VAL_DEV,     // MATRIX A(CSC)
                c_jdx_d, c_ptr_d, c_val_d,                  // MATRIX C(CSR)
                RUPP_DEV,                                   // UPPER BOUND           
                WLS_DEV,                                    // WLS_DEV
                RBOX_DEV);
        cudaDeviceSynchronize();
        ERROR_CHECK;
    }
    for(int i=0;i<8;i++){
        printf("%d ",thread_counter_h[i]);
    }
    printf("\n");
/*    float total_eff = 0;

    for(int i=0;i<8;i++){
        int stripe;
        int base;
        int box;
        if(i==0){ stripe = 32; box=64;} 
        else if(i==1) {stripe=16; box = 64;}
        else if(i==2) {stripe=8; box = 64;} 
        else if(i==3) {stripe=4; box = 64;}
        else if(i==4) {stripe=2; box = 64;}
        else {stripe = 1; box = 128;}

        if(i==0) base = 0;
        else thread_bin_h[i-1];

        float eff = 0;
        for(int j=0;j<thread_counter_h[i];j+=stripe){
            float temp = 0;
            int max = 1;
            for(int k=0;k<stripe;k++){
                if(thread_counter_h[i] <= j+k) break;
                int index = idx_bin_h[base+j+k];
                int len = CSR_HOST.header[index+1] - CSR_HOST.header[index];
                if(max<len) max = len;
                temp +=len*(CSC_HOST.header[index+1] - CSC_HOST.header[index]);
            }
            temp/=max;
            temp/=box;
            eff += temp;
        }
        printf("%f %d %d\n",eff, thread_counter_h[i], thread_bin_h[8]);
        total_eff += eff/((float)thread_bin_h[8]);

    }
    printf("%f \n",total_eff);
*/


       cudaDeviceSynchronize();
       cudaEventRecord(ev_spgemm_e, 0);
       cudaEventSynchronize(ev_spgemm_e);
    cudaEventElapsedTime(&t_spgemm_l,ev_spgemm_s, ev_spgemm_e);

}

void cudaPassBB(int n,int t){	
    int BLOCK_SIZE = 384;

	dim3 block(BLOCK_SIZE);
    dim3 grid(t);
    cudaEvent_t ev_spgemm_s, ev_spgemm_e;
    cudaEventCreate(&ev_spgemm_s);
    cudaEventCreate(&ev_spgemm_e);
    cudaEventRecord(ev_spgemm_s, 0);

    cudaMalloc((void**)&(P_DEV), sizeof(PP)*t); ERROR_CHECK;
    cudaMemcpy((void*)P_DEV, (const void*)P_HOST,  sizeof(PP)*(t), cudaMemcpyHostToDevice); ERROR_CHECK;
    cudaMemcpy((void*)CSR_CIDX_DEV, (const void*) DCSR_HOST.cidx, sizeof(int)*DCSR_HOST.e, cudaMemcpyHostToDevice); ERROR_CHECK;
    cudaMemcpy((void*)CSR_VAL_DEV, (const void*) DCSR_HOST.val, sizeof(float)*DCSR_HOST.e, cudaMemcpyHostToDevice); ERROR_CHECK;
    cudaMemcpy((void*)CSR_PTR_DEV, (const void*) DCSR_HOST.header, sizeof(int)*(DCSR_HOST.n+1), cudaMemcpyHostToDevice); ERROR_CHECK;
    cudaMemcpy((void*)CSC_RIDX_DEV, (const void*) DCSC_HOST.ridx, sizeof(int)*DCSC_HOST.e, cudaMemcpyHostToDevice); ERROR_CHECK;
    cudaMemcpy((void*)CSC_VAL_DEV, (const void*) DCSC_HOST.val, sizeof(float)*DCSC_HOST.e, cudaMemcpyHostToDevice); ERROR_CHECK;
	cudaMemcpy((void*)CSC_PTR_DEV, (const void*) DCSC_HOST.header, sizeof(int)*(DCSC_HOST.n+1), cudaMemcpyHostToDevice); ERROR_CHECK;

    
    calcInterDom<<<grid, block>>>(
            CSR_CIDX_DEV, CSR_PTR_DEV, CSR_VAL_DEV,     // MATRIX B(CSR)
            CSC_RIDX_DEV, CSC_PTR_DEV, CSC_VAL_DEV,     // MATRIX A(CSC)
            c_jdx_d, c_ptr_d, c_val_d,
            //COO_DEV,                                  // INTERMEDIATE C(COO)
            RUPP_DEV,                                   // UPPER BOUND           
            P_DEV,                                    // WLS_DEV
            RBOX_DEV);                                  // RBOX
            ERROR_CHECK;

   	cudaEventRecord(ev_spgemm_e, 0);
	cudaEventSynchronize(ev_spgemm_e);
    cudaEventElapsedTime(&t_spgemm_d,ev_spgemm_s, ev_spgemm_e);
}

void cudaPassC(int n){	
    int BLOCK_SIZE = MT;

    int GRID_SIZE = GRID;
    float* DROW_DEV;
    
    cudaMalloc((void**) &(DROW_DEV), sizeof(float)*n*GRID_SIZE);  ERROR_CHECK; 
    cudaMemset(DROW_DEV, 0, sizeof(float)*n*GRID_SIZE); ERROR_CHECK;

//  cudaMallocHost((void**)&(c_val_h),sizeof(float)*(RUPP_HOST[n]));
//    cudaMallocHost((void**)&(c_idx_h),sizeof(int)*(RUPP_HOST[n]));
    c_val_h=(float*)malloc(sizeof(float)*RUPP_HOST[n]);
    c_idx_h=(int*)malloc(sizeof(int)*RUPP_HOST[n]);

    nnzC = 0;
        dim3 block(BLOCK_SIZE);
        dim3 grid(GRID);
		cudaEvent_t ev_merge_s, ev_merge_e;
		cudaEventCreate(&ev_merge_s);
		cudaEventCreate(&ev_merge_e);
		cudaEventRecord(ev_merge_s, 0);
        merge<<<grid, block>>>(
                RUPP_DEV,                                   // UPPER BOUND           
                RBOX_DEV,                                   // RBOX
                DROW_DEV,                                   // DENSE
                c_jdx_d,c_val_d,c_ptr_d,
                c_idx_d,
                else_c,
                number_of_else_in_c,
                N_DEV);                                 
        cudaDeviceSynchronize();
        merge_limitting<<<grid, block>>>(
                RUPP_DEV,                                   // UPPER BOUND           
                RBOX_DEV,                                   // RBOX
                DROW_DEV,                                   // DENSE
                c_jdx_d,c_val_d,c_ptr_d,
                c_idx_d,
                dominator_c,
                number_of_dominators_in_c,
                N_DEV);                                  

    cudaDeviceSynchronize();
    cudaEventRecord(ev_merge_e, 0);
	cudaEventSynchronize(ev_merge_e);
    cudaEventElapsedTime(&t_merge,ev_merge_s, ev_merge_e);
    printf("%d\n",number_of_dominators_in_c_h[0]);

    c_ptr_h = (int*)malloc(sizeof(int)*(n+1));
    c_ptr_h[0] = 0;
    cudaMemcpy((void*)&c_ptr_h[1],(const void*) &c_ptr_d[0], sizeof(int)*n , cudaMemcpyDeviceToHost);
//    cudaMemcpyAsync((void*)&c_val_h[0],(const void*) &c_val_d[0], sizeof(float)*RUPP_HOST[n] , cudaMemcpyDeviceToHost);
//    cudaMemcpyAsync((void*)&c_idx_h[0],(const void*) &c_idx_d[0], sizeof(int)*RUPP_HOST[n], cudaMemcpyDeviceToHost);


    int sHost = 0;
    for(int i=0;i<n;i++){
        int sDev = RUPP_HOST[i];
        int l = c_ptr_h[i+1];
        //cudaDeviceSynchronize();
        nnzC += l;
        //cudaMemcpy((void*)&MVAL_HOST[sHost],(const void*) &MVAL_DEV[sDev], sizeof(float)*l , cudaMemcpyDeviceToHost);
        //cudaThreadSynchronize();
        //cudaMemcpyAsync((void*)&MIDX_HOST[sHost],(const void*) &MIDX_DEV[sDev], sizeof(int)*l, cudaMemcpyDeviceToHost);
        cudaMemcpyAsync((void*)&c_val_h[sHost],(const void*) &c_val_d[sDev], sizeof(float)*l , cudaMemcpyDeviceToHost);
        cudaMemcpyAsync((void*)&c_idx_h[sHost],(const void*) &c_idx_d[sDev], sizeof(int)*l, cudaMemcpyDeviceToHost);

  
        
        sHost +=l;
        //MPTR_HOST[i+1] += MPTR_HOST[i];
        c_ptr_h[i+1] += c_ptr_h[i];
    }

 /*  
    for(int j=0;j < RUPP_HOST[n] ;j++){
        if(c_val_h[j]>0.000001 ){//|| MVAL_HOST[j]<-0.00001){
            nnzC++;
        }
    }
*/
}


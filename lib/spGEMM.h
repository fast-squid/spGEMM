#ifndef SPGEMM_H_
#define SPGEMM_H_

#ifdef __cplusplus
extern "C" {
#endif
#include "coo.h"
#include "csc.h"
#include "csr.h"
#include "wlt.h"

struct _PP{
    int r;
    int c;
};
typedef struct _PP PP;

extern int subMatrixSize;
extern int depth;
extern int resCnt;

extern coo COOC;
extern coo COOR;


///////////////// DEVICE //////////////////
extern int* CSC_RIDX_DEV;       //
extern int* CSC_PTR_DEV;        // MATRIX A
extern float* CSC_VAL_DEV;      //

extern int* CSR_CIDX_DEV;       // 
extern int* CSR_PTR_DEV;        // MATRIX B
extern float* CSR_VAL_DEV;      //


extern int* RUPP_DEV;
extern int* WLS_DEV;

extern int* RBOX_DEV;

extern PP* P_DEV;
extern int* N_DEV;
////////////////// HOST //////////////////

extern int* WLS_HOST;
extern int* RUPP_HOST;

extern PP* P_HOST;


/////////////////////////////////////////
extern int* c_jdx_d;
extern int* c_idx_d;
extern float* c_val_d;
extern int* c_ptr_d;
extern int* c_idx_h;
extern float* c_val_h;
extern int* c_ptr_h;
////////////////////////////////////////

//extern int* number_of_dominators_in_c;
//extern int* dominator_c;

extern coo COO_HOST;
extern csc CSC_HOST;
extern csr CSR_HOST;
extern csr DCSR_HOST;
extern csc DCSC_HOST;


extern float t_pre, t_fd,t_spgemm_l,t_spgemm_d,t_merge,t_bin;
extern double t_cpu;
extern int nnzC;
extern int *domR;
extern int *domN;
extern int *domRC;
extern int *domCC;
extern int domRCavg;
extern int domCCavg;

void cudaPass_(int, int);
void cudaPath_find_thread_count_approx(int n);

void cudaPass_F(int);
void cudaPassB(int);
void cudaPassBB(int,int);
void cudaPassC(int);

void cudaPass0(int, int);
int cudaPass2();
int spGEMM(csc CSC, csr CSR, wlt WLT, wlt_cheat WLT_C,int cheat);
#ifdef __cplusplus
}
#endif

#endif //SPARSE_mat_FORM_H_


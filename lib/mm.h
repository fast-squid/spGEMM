#ifndef SPGEMM_H_
#define SPGEMM_H_

#ifdef __cplusplus
extern "C" {
#endif
#include "coo.h"
#include "cm.h"

struct DenseChunk
{
    int idx;
    float val[64];
};
typedef struct DenseChunk dc;

cm cudaCOO2CM(coo mat);
cm cudaInitGEMM(cm A, cm B);
void cudaCategorizeGEMM(cm A, cm B);
void cudaSimpleGEMM(cm A, cm B, cm C);
void cudaSplitGEMM(cm A, cm B, cm C);
void cudaBinGEMM(cm A, cm B, cm C);
void cudaGatherGEMM(cm A, cm B, cm C);

void cudaDCGEMM(cm A, cm B, cm C);
//void cudaMergeGEMM(cm A, cm B, cm C);
void cudaMergeGEMM(cm C);

//void test(cm A, cm B);
#ifdef __cplusplus
}
#endif

#endif //SPARSE_mat_FORM_H_


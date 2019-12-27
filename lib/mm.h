#ifndef SPGEMM_H_
#define SPGEMM_H_

#ifdef __cplusplus
extern "C" {
#endif
#include "coo.h"
#include "cm.h"

cm cudaCOO2CM(coo mat);
void cudaInitGEMM(cm A, cm B);
//void test(cm A, cm B);
#ifdef __cplusplus
}
#endif

#endif //SPARSE_mat_FORM_H_

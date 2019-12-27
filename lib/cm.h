#ifndef CSR_H_
#define CSR_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "coo.h"

struct CompressedMatrix
{
	int *ptr;
	int *idx;
	float *val;

	int nnz;
	int row_size;
    int col_size;
    bool type;
};
typedef struct CompressedMatrix cm;

void initCM( cm* mat );
void freeCSR( cm* mat );
void setCMtype(cm* mat, bool type);
void printCM(cm mat);
//cm transformCOOToCSR( coo mat );

#ifdef __cplusplus
}
#endif

#endif //SPARSE_mat_FORM_H_

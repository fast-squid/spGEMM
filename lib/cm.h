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
	int num_rows;
    int num_cols;
    bool type;
};
typedef struct CompressedMatrix cm;

void cmInit(cm* mat);
void cmFree(cm* mat);
int cmGetNumRows(cm mat);
int cmGetNumCols(cm mat);
int cmGetNNZ(cm mat);
void cmSetNumRows(cm* mat, int num_rows);
void cmSetNumCols(cm* mat, int num_cols);
void cmSetNNZ(cm* mat, int nnz);
void cmSetType(cm* mat,bool type);


#ifdef __cplusplus
}
#endif

#endif //SPARSE_mat_FORM_H_

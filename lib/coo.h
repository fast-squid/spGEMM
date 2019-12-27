#ifndef COO_H_
#define COO_H_

#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>

#define COO_IDX_FORM 2
#define COO_IDX_VALUE_FORM 3
#define SORT_COO_ROW_BASE 0
#define SORT_COO_COL_BASE 1
#define ROW_MAJOR 0
#define COL_MAJOR 1

struct triplet
{
	int ridx;
	int cidx;
	float val;
};
typedef struct triplet triplet;

struct coo
{
	triplet* data;

	int nnz;
	int row_size;
    int col_size;
    bool type;
};
typedef struct coo coo;

void initCOO(coo* mat);
void freeCOO(coo* mat);
void setCOOtype(coo* mat,bool type);
coo copyCOO(coo mat);
void sortCOO(coo mat);

void readCOO(coo* mat, char* path);
void readMTX(coo* mat, char* path);

#ifdef __cplusplus
}
#endif

#endif //SPARSE_mat_FORM_H_

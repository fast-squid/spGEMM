#ifndef WLT_H_
#define WLT_H_

#ifdef __cplusplus
extern "C" {
#endif
#include "csc.h"
#include "csr.h"

struct wltData
{
    int idx;
    int base_idx;
	int num_of_row_elements;
	int num_of_col_elements;
};
typedef struct wltData wltData;

struct wlt
{
	wltData* data;
	int threadblocks;
	int workloads;
};
typedef struct wlt wlt;

struct wltData_cheat
{
    int ridx;
    int cidx;
    int base_idx;
	int num_of_row_elements;
	int num_of_col_elements;
};
typedef struct wltData_cheat wltData_cheat;

struct wlt_cheat
{
	wltData_cheat* data;
	int threadblocks;
	int workloads;
};
typedef struct wlt_cheat wlt_cheat;

/*
extern wltData *idxWLT;
extern int totalWL;
*/
extern int maxWL;

void buildIdxWLT(int n);
void freeWLT(wlt* table );
void freeWLTC(wlt_cheat* table );

int wltDataCompare( const void* a, const void* b );
int wltDataCompare_cheat( const void* a, const void* b );
wlt makeWLT( csc cscMat, csr csrMat);
wlt_cheat makeWLT_cheat( csc cscMat, csr csrMat,int,int,int);
coo extractCOOR(coo mat, wltData* WLTD,int);
coo extractCOOC(coo mat, wltData* WLTD,int);




#ifdef __cplusplus
}
#endif

#endif //SPARSE_MATRIX_FORM_H_

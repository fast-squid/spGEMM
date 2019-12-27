#ifndef CSC_H_
#define CSC_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "coo.h"

struct csc
{
	int *header;
	int *ridx;
	float *val;

	int n;
	int e;
};
typedef struct csc csc;

void initCSC( csc* mat);
void freeCSC( csc* mat);
void dumpCSC( csc matrix);
csc transformCOOToCSC( coo mat );

#ifdef __cplusplus
}
#endif

#endif 

#include <stdio.h>
#include <stdlib.h>
#include "../lib/coo.h"
#include "../lib/cm.h"


void initCM( cm* mat )
{
	mat->ptr = NULL;
	mat->idx = NULL;
	mat->val = NULL;
	mat->nnz = 0;
	mat->row_size = 0;
    mat->col_size = 0;
    mat->type = ROW_MAJOR;
}

void freeCSR(cm* mat)
{
    free( mat->ptr );
    free( mat->idx );
    free( mat->val );
    mat->ptr = NULL;
    mat->idx = NULL;
 	mat->val = NULL;

	mat->nnz = 0;
	mat->row_size = 0;
    mat->col_size = 0;
    mat->type = ROW_MAJOR;
}


void setCMtype(cm* mat,bool type)
{
    mat->type = type;
}

void printCM(cm mat)
{
    int outer_iter_size;
    int inner_iter_size;

    if(mat.type == ROW_MAJOR){
        outer_iter_size = mat.row_size;
        printf("CSR %d %d %d\n",mat.row_size,mat.col_size,mat.nnz);
    }
    else{
        outer_iter_size = mat.col_size;
        printf("CSC %d %d %d\n",mat.row_size,mat.col_size,mat.nnz);
    }

    int cnt = 0;
    for(int i=0;i<outer_iter_size;i++)
    { 
        inner_iter_size = mat.ptr[i+1] - mat.ptr[i];
        int base = mat.ptr[i];
        for(int j=0;j< inner_iter_size; j++)
        {
            printf("%d\t%d\t%d\t%f\n",cnt++, i,mat.idx[base + j],mat.val[base + j]);
        }
    }
}


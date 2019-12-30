#include <stdio.h>
#include <stdlib.h>
#include "../lib/coo.h"
#include "../lib/cm.h"


void cmInit( cm* mat )
{
	mat->ptr = NULL;
	mat->idx = NULL;
	mat->val = NULL;
	mat->nnz = 0;
	mat->num_cols = 0;
    mat->num_rows = 0;
    mat->type = ROW_MAJOR;
}

void cmFree(cm* mat)
{
    free( mat->ptr );
    free( mat->idx );
    free( mat->val );
    mat->ptr = NULL;
    mat->idx = NULL;
 	mat->val = NULL;

	mat->nnz = 0;
	mat->num_cols = 0;
    mat->num_rows = 0;
    mat->type = ROW_MAJOR;
}


int cmGetNumRows(cm mat)
{
    return mat.num_rows;
}
int cmGetNumCols(cm mat)
{
    return mat.num_cols;
}
int cmGetNNZ(cm mat)
{
    return mat.nnz;
}

void cmSetNumRows(cm* mat, int num_rows)
{
    mat->num_rows = num_rows;
}
void cmSetNumCols(cm* mat, int num_cols)
{
    mat->num_cols = num_cols;
}
void cmSetNNZ(cm* mat, int nnz)
{
    mat->nnz = nnz;
}
void cmSetType(cm* mat,bool type)
{
    mat->type = type;
}


void printCM(cm mat)
{
    int outer_iter_size;
    int inner_iter_size;

    if(mat.type == ROW_MAJOR){
        outer_iter_size = mat.num_cols;
        printf("CSR %d %d %d\n",mat.num_cols,mat.num_rows,mat.nnz);
    }
    else{
        outer_iter_size = mat.num_rows;
        printf("CSC %d %d %d\n",mat.num_cols,mat.num_rows,mat.nnz);
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


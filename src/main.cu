#include <stdio.h>
#include <stdlib.h>
#include "../lib/coo.h"
#include "../lib/cm.h"
#include "../lib/mm.h"

#define ITER 1

/* EXTERN DEVICE POINTER */

int main(int argc, char *argv[]){
	cudaFree(0);
    int* init = new int;
    cudaMalloc((void**)&init,sizeof(int)); 

    for(int tc =0;tc<ITER;tc++){
        coo t_A,t_B;
        initCOO(&t_A);    
        initCOO(&t_B);    
        setCOOtype(&t_A,COL_MAJOR);
        setCOOtype(&t_B,ROW_MAJOR);
        
        readMTX(&t_A, argv[1]);  
        readMTX(&t_B, argv[2]);
        
        printf("sorting A..\n");
        sortCOO(t_A);
        printf("sorting B..\n");
        sortCOO(t_B);
/*       
        printf("%d %d\n",t_A.nnz, t_B.nnz);        
        for(int i=0;i<t_A.nnz;i++)
        {
            printf("%d %d\n",t_A.data[i].ridx, t_A.data[i].cidx);
        }
*/
        cm A = cudaCOO2CM(t_A);
        cm B = cudaCOO2CM(t_B);
        cm C = cudaInitGEMM(A,B);
        cudaCategorizeGEMM(A,B);
        cudaSplitGEMM(A,B,C);
        cudaGatherGEMM(A,B,C);
        //cudaBinGEMM(A,B,C);
        //cudaSimpleGEMM(A,B,C);
        cudaMergeGEMM(C);
        //test(csc, csr);
        
        freeCOO(&t_A);
        freeCOO(&t_B);
    }
    
   	return 0;
}


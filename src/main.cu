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
        coo A,B;
        initCOO(&A);    
        initCOO(&B);    
        setCOOtype(&A,COL_MAJOR);
        setCOOtype(&B,ROW_MAJOR);
        
        readMTX(&A, argv[1]);  
        readMTX(&B, argv[1]);
        
        printf("sorting A..\n");
        sortCOO(A);
        printf("sorting B..\n");
        sortCOO(B);
       
        cm csc = cudaCOO2CM(A);
        cm csr = cudaCOO2CM(B);
//        printCM(csc);
//        printCM(csr);
        cudaInitGEMM(csc,csr);
        //test(csc, csr);
        
        freeCOO(&A);
        freeCOO(&B);
        return 0;
    }
    
   	return 0;
}


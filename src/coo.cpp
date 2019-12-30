#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../lib/coo.h"

void initCOO( coo* mat )
{
    mat->data= NULL;
	mat->nnz = 0;
	mat->num_cols = 0;
    mat->num_rows = 0;
    mat->type = ROW_MAJOR;
}

void freeCOO( coo* mat )
{
	free(mat->data);
	mat->nnz = 0;
	mat->num_cols = 0;
    mat->num_rows = 0;
    mat->type = ROW_MAJOR;
}

coo copyCOO( coo mat )
{
	coo ret;

    ret.nnz = mat.nnz;
    ret.num_cols = mat.num_cols;
    ret.num_rows = mat.num_rows;
	
    ret.data = new triplet[ret.nnz];
	memcpy( ret.data, mat.data, sizeof(triplet) * ret.nnz );
	
	return ret;
}

void setCOOtype(coo* mat, bool type)
{
    mat->type = type;
}

int cooCompareByRow( const void* a, const void* b)
{
	triplet dataA = *(triplet*)a;
	triplet dataB = *(triplet*)b;

	if( dataA.ridx != dataB.ridx )
		return dataA.ridx - dataB.ridx;
	else if( dataA.cidx != dataB.cidx )
		return dataA.cidx - dataB.cidx;
	else if( dataA.val > dataB.val )
		return 1;
	else
		return -1;
}

int cooCompareByCol( const void* a, const void* b)
{
	triplet dataA = *(triplet*)a;
	triplet dataB = *(triplet*)b;

	if( dataA.cidx != dataB.cidx )
		return dataA.cidx - dataB.cidx;
	else if( dataA.ridx != dataB.ridx )
		return dataA.ridx - dataB.ridx;
	else if( dataA.val > dataB.val )
		return 1;
	else
		return -1;
}

void sortCOO( coo mat )
{
	if( mat.type == ROW_MAJOR )
		qsort( (void*)mat.data, (size_t)mat.nnz, sizeof(triplet), cooCompareByRow );
	if( mat.type == COL_MAJOR )
		qsort( (void*)mat.data, (size_t)mat.nnz, sizeof(triplet), cooCompareByCol );
}


void readCOO( coo* mat, char*path)
{
	FILE* fp = fopen(path,"r");
	int dim = 0;

	fscanf( fp, "%d", &(dim));
	fscanf( fp, "%d", &(mat->num_cols));
	fscanf( fp, "%d", &(mat->nnz));

    mat->num_rows = mat->num_cols;

	mat->data  = new triplet[mat->nnz];
    for( int i = 0; i<mat->nnz; i++ ){
        int ridx, cidx;

    	fscanf( fp, "%d", &ridx);
   		fscanf( fp, "%d", &cidx);
        mat->data[i].ridx = ridx;
        mat->data[i].cidx = cidx;

        if(dim==2)
            mat->data[i].val=1.1;
        else
            fscanf(fp,"%f",&mat->data[i].val);
   }
    fclose(fp);
}

void readMTX( coo* mat, char*path )
{
	FILE* fp = fopen(path,"r");
	int dim = 0;

    char line[100];
    int sym = 0;
    // read banner
    fgets(line,100,fp);
    if(strstr(line,"symmetric")) sym = 1;
    while(1){
        fgets(line,100,fp);
        if(line[0]!='%'){
            break;
        }
    }

    mat->nnz = 0;
	mat->num_cols = atoi(strtok(line," \n\t"));
    mat->num_rows = atoi(strtok(NULL," \n\t"));
    int max_nnz = atoi(strtok(NULL," \n\t"));
     
	if(sym){
        mat->data = new triplet[max_nnz*2];
    }
    else{
        mat->data = new triplet[max_nnz];
    }

    int i=0;
    int r,c;
    float val;
    char dummy[200];
    printf("%d %d %d\n",mat->nnz, mat->num_cols, mat->num_rows);
    while(fscanf(fp,"%d %d %s ",&r,&c,dummy)!=EOF){
    //while(fscanf(fp,"%d %d %s %s",&r,&c,fuck,fuck)!=EOF){

        if(sym){
            if(r==c){
                mat->data[i].ridx = r-1;
                mat->data[i].cidx = c-1;
                mat->data[i].val = 1.1;
                mat->nnz++; i++;
            }
            else{
                mat->data[i].ridx = r-1;
                mat->data[i].cidx = c-1;
                mat->data[i].val = 1.1;
                mat->data[i+1].ridx = c-1;
                mat->data[i+1].cidx = r-1;
                mat->data[i+1].val = 1.1;

                mat->nnz+=2; i+=2;
            }
        }
        else{
            mat->data[i].ridx = r-1;
            mat->data[i].cidx = c-1;
            mat->data[i].val = 1.1;
            mat->nnz++; i++;
        }
    }
    fclose(fp);
}





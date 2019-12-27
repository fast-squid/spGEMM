#ifndef THREAD_H_
#define THREAD_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <pthread.h>
#include "coo.h"

extern pthread_mutex_t t_lock;

struct T_PARAMETERS{
	int tid;
	int t_cnt;
	pthread_barrier_t* t_barrier;

	coo* mat[2];
};
typedef struct T_PARAMETERS T_PARAMETERS;
double get_time();
void createThread(void *(func)(void*), int t_cnt);
void* dddd(void* data);
void* eeee(void* data);

void* sort(void* data);
void multiplication();
void* pre(void* data);
void* reduction(void* data);

#ifdef __cplusplus
}
#endif

#endif //SPARSE_MATRIX_FORM_H_

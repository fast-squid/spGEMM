CC=gcc
CXX=g++
NVCC=nvcc

CFLAGS:= -W -Wall -g -c -I -O3 ./lib -std=gnu99 #-lpthread 
CPPFLAGS:= -W -Wall -g -c -I ./lib
NVCCFLAGS:= -g -c -I ./lib -arch=sm_61 



all : ./obj ./bin ./bin/outer ./bin/count  

clean:
	rm -vf ./obj/*.o
	rm -vf ./bin/*

./obj :
	mkdir ./obj

./bin :
	mkdir ./bin

./obj/cm.cpp.o : ./src/cm.cpp 
	$(CXX) $(CPPFLAGS) -o $@ $<
./obj/coo.cpp.o : ./src/coo.cpp 
	$(CXX) $(CPPFLAGS) -o $@ $<
./obj/mm.cu.o : ./src/mm.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

#./obj/csr.c.o : ./src/csr.c ./lib/csr.h 
#	$(CC) $(CFLAGS) -o $@ $<
#
#./obj/csc.c.o : ./src/csc.c ./lib/csc.h 
#	$(CC) $(CFLAGS) -o $@ $<
#
#./obj/thread.c.o : ./src/thread.c ./lib/coo.h  ./lib/csr.h ./lib/csc.h ./lib/thread.h ./lib/spGEMM.h
#	$(CC) $(CFLAGS) -o $@ $<
#	
#./obj/count.c.o : ./src/count.c
#	$(CC) $(CFLAGS) -o $@ $<
#
#./obj/heap.c.o : ./src/heap.c ./lib/coo.h ./lib/wlt.h
#	$(CC) $(CFLAGS) -o $@ $<
#



#./obj/spGEMM.cu.o : ./src/spGEMM.cu ./lib/coo.h ./lib/csr.h ./lib/csc.h ./lib/wlt.h 
#	$(NVCC) $(NVCCFLAGS) --ptxas-options=-v -o $@ $<
#
./obj/main.cu.o : ./src/main.cu  
	$(NVCC) $(NVCCFLAGS) -o $@ $<

./bin/outer : ./obj/main.cu.o ./obj/coo.cpp.o ./obj/cm.cpp.o ./obj/mm.cu.o 
	nvcc  -arch=sm_61  -o $@ $^	
	










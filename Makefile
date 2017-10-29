a.out: SCC.cu
	nvcc SCC.cu -g -G -arch=sm_20
clean:
	rm a.out

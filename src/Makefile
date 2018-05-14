
CFLAGS = -g -O2 -Wall -fopenmp -funroll-loops -ffast-math
LDFLAGS = -fopenmp
LDLIBS = -lm -lhdf5_serial -lfftw3

OBJS = main.o grid.o hdf5.o perf.o

grid : $(OBJS)
recombine : recombine.o test_recombine.o

.PHONY: clean
clean :
	rm $(OBJS)

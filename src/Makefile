
CFLAGS += -g -Wall -fopenmp -ffast-math
LDFLAGS += -fopenmp -g
LDLIBS = -L$(subst :, -L,$(LIBRARY_PATH)) -lm -lhdf5 -lfftw3
CC = mpicc

GRID_OBJS = main.o grid.o hdf5.o perf.o
RECOMBINE_OBJS = recombine.o test_recombine.o bench_recombine.o

grid : $(GRID_OBJS)
recombine : $(RECOMBINE_OBJS)

.PHONY: clean
clean :
	rm -f $(GRID_OBJS)
	rm -f $(RECOMBINE_OBJS)

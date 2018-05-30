
CFLAGS += -g -Wall -fopenmp -ffast-math
LDFLAGS += -fopenmp -g
LDLIBS = -L$(subst :, -L,$(LIBRARY_PATH)) -lm -lhdf5 -lfftw3
CC = mpicc

GRID_OBJS = main.o grid.o hdf5.o perf.o
TEST_RECOMBINE_OBJS = recombine.o test_recombine.o
BENCH_RECOMBINE_OBJS = recombine.o bench_recombine.o
CONFIG_OBJS = test_config.o hdf5.o

grid : $(GRID_OBJS)
test_recombine : $(TEST_RECOMBINE_OBJS)
recombine : $(BENCH_RECOMBINE_OBJS)
test_config : $(CONFIG_OBJS)

.PHONY: clean
clean :
	rm -f $(GRID_OBJS) $(RECOMBINE_OBJS) $(CONFIG_OBJS)

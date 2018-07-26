
HDF5_INCLUDE=/usr/include/hdf5/serial
HDF5_LIB=/usr/lib/x86_64-linux-gnu/hdf5/serial

CFLAGS += -g -Wall -fopenmp -ffast-math -I$(HDF5_INCLUDE)
LDFLAGS += -fopenmp -g
LDLIBS = -L$(HDF5_LIB) -lm -lhdf5 -lfftw3
CC = mpicc

GRID_OBJS = main.o grid.o hdf5.o perf.o
TEST_RECOMBINE_OBJS = recombine.o test_recombine.o grid.o hdf5.o
BENCH_RECOMBINE_OBJS = recombine.o bench_recombine.o hdf5.o config.o
CONFIG_OBJS = test_config.o config.o recombine.o  hdf5.o

grid : $(GRID_OBJS)
test_recombine : $(TEST_RECOMBINE_OBJS)
recombine : $(BENCH_RECOMBINE_OBJS)
test_config : $(CONFIG_OBJS)

.PHONY: clean
clean :
	rm -f $(GRID_OBJS) $(BENCH_RECOMBINE_OBJS) $(TEST_RECOMBINE_OBJS) $(CONFIG_OBJS) grid test_recombine recombine test_config


HDF5_INC ?= /usr/include/hdf5/serial
HDF5_LIB ?= /usr/lib/x86_64-linux-gnu/hdf5/serial

CFLAGS += -Wall -fopenmp -ffast-math -I$(HDF5_INC) -ggdb -march=native -O2
LDFLAGS += -fopenmp -ggdb -O2
LDLIBS = -L$(HDF5_LIB) -lm -lhdf5 -lfftw3
CC = mpicc

IOTEST_OBJS = iotest.o recombine.o hdf5.o config.o producer.o streamer.o grid.o
TEST_RECOMBINE_OBJS = recombine.o test_recombine.o grid.o hdf5.o
TEST_CONFIG_OBJS = test_config.o config.o recombine.o  hdf5.o

iotest : $(IOTEST_OBJS)
recombine : $(BENCH_RECOMBINE_OBJS)
test_config : $(TEST_CONFIG_OBJS)

.PHONY: clean
clean :
	rm -f $(IOTEST_OBJS) $(TEST_RECOMBINE_OBJS) $(TEST_CONFIG_OBJS) grid test_recombine recombine test_config

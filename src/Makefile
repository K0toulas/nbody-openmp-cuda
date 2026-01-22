# CUDA Makefile 
CC      = nvcc
CFLAGS  = -O3 -use_fast_math -Xcompiler -fopenmp
SRC     = nbody.cu
EXE     = nbody

all: $(EXE)

$(EXE): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(EXE)

clean:
	rm -f $(EXE)

.PHONY: all clean
# Build for the GPU batch image-processing pipeline.
#
#   make            build bin/gpu_batch_pipeline
#   make run        build and run on data/input with default settings
#   make clean      remove build products
#
# Override the target GPU architectures if needed, e.g.
#   make GENCODE="-gencode arch=compute_75,code=sm_75"

NVCC     ?= nvcc
BIN_DIR  := bin
OBJ_DIR  := build
TARGET   := $(BIN_DIR)/gpu_batch_pipeline

GENCODE  ?= -gencode arch=compute_60,code=sm_60 \
            -gencode arch=compute_70,code=sm_70 \
            -gencode arch=compute_75,code=sm_75 \
            -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 \
            -gencode arch=compute_86,code=compute_86

# -ffp-contract=off keeps the CPU reference free of fused multiply-adds so it
# matches the GPU kernels bit for bit.
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo $(GENCODE) \
             -Xcompiler -Wall,-Wextra,-ffp-contract=off

SOURCES  := $(wildcard src/*.cu src/*.cc)
OBJECTS  := $(patsubst src/%,$(OBJ_DIR)/%.o,$(SOURCES))
HEADERS  := $(wildcard src/*.h src/*.cuh)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

$(OBJ_DIR)/%.cu.o: src/%.cu $(HEADERS) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(OBJ_DIR)/%.cc.o: src/%.cc $(HEADERS) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

run: $(TARGET)
	./run.sh

clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR)

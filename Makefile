NVCC      ?= nvcc
MPICXX    ?= mpicxx

GENCODES := \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_80,code=compute_80 \
  -gencode arch=compute_90,code=compute_90

NVCCBASE := -O3 -std=c++17 $(GENCODES)
INCLUDES := -I libarff
LIBARFF_SRCS := \
  libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp \
  libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp \
  libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp

TARGETS := cuda cuda-mpi cuda-single

all: $(TARGETS)

cuda cuda-single: % : %.cu $(LIBARFF_SRCS)
	$(NVCC) $(NVCCBASE) -ccbin=$(CXX) $(INCLUDES) $^ -o $@

cuda-mpi: cuda-mpi.cu $(LIBARFF_SRCS)
	$(NVCC) $(NVCCBASE) -ccbin=$(MPICXX) $(INCLUDES) $^ -o $@

serial: serial.cpp
	g++ -std=c++11 -o serial serial.cpp $(INCLUDES) $(LIBARFF_SRCS)

clean:
	rm -f serial $(TARGETS)

.PHONY: all clean


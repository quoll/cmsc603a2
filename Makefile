NVCC      ?= nvcc
MPICXX    ?= mpicxx

GENCODES := \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_80,code=compute_80 \
  -gencode arch=compute_90,code=compute_90

NVCCFLAGS := -O3 -std=c++17 $(GENCODES) -ccbin=$(MPICXX)
INCLUDES  := -I libarff
LIBARFF_SRCS := \
  libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp \
  libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp \
  libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp

TARGETS   := cuda cuda-mpi
CU_SRCS   := cuda.cu cuda-mpi.cu

all: $(TARGETS)

serial: serial.cpp
	g++ -std=c++11 -o serial serial.cpp -I libarff libarff/arff_attr.cpp libarff/arff_data.cpp libarff/arff_instance.cpp libarff/arff_lexer.cpp libarff/arff_parser.cpp libarff/arff_scanner.cpp libarff/arff_token.cpp libarff/arff_utils.cpp libarff/arff_value.cpp

$(TARGETS): % : %.cu $(LIBARFF_SRCS)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@

clean:
	rm -f serial $(TARGETS)

.PHONY: all clean


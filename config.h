#ifndef __CNN_HLS_CONFIG_H__
#define __CNN_HLS_CONFIG_H__
#ifndef __VITIS_HLS__
#define __VITIS_HLS__
#endif

#define BATCH_SIZE 8
#define VALUE_SIZE 32
#define VALUE_FLOAT 0
#define VALUE_INT_PART 8
#define INT_SIZE 32
#define INT_PRIMITIVE 1
#define PACK_W 5
#define LAYER6_TANH_CPU 1
#define REUSE_LAYER_FUNCTIONS 1
#define CONV_DATAFLOW 1

#endif

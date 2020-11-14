#ifndef __CNN_HLS_CONFIG_H__
#define __CNN_HLS_CONFIG_H__
#ifndef __VITIS_HLS__
#define __VITIS_HLS__
#endif

#define BATCH_SIZE 16
typedef ap_fixed<32, 8> value_t;
//typedef ap_fixed<64, 8> value_t;
//typedef float value_t;
typedef hlslib::DataPack<value_t, BATCH_SIZE> minibatch_t;

#endif

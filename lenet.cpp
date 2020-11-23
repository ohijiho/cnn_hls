#include "config.h"
#include "cnn_functions.h"
#include "utils.h"

void lenet1(
		const minibatch_t *__restrict layer1_x, minibatch_t *__restrict layer1_y, const weight_t *__restrict layer1_weight, const weight_t *__restrict layer1_bias,
		size2_t layer1_input_size,
		uint_t layer1_in_channels, uint_t layer1_out_channels,
		size2_t layer1_kernel_size, size2_t layer1_stride, size2_t layer1_padding, size2_t layer1_dilation,

		const minibatch_t *__restrict layer2_x, minibatch_t *__restrict layer2_y,
		uint_t layer2_channels, size2_t layer2_input_size,
		size2_t layer2_kernel_size, size2_t layer2_stride, size2_t layer2_padding, size2_t layer2_dilation,

		const minibatch_t *__restrict layer3_x, minibatch_t *__restrict layer3_y, const weight_t *__restrict layer3_weight, const weight_t *__restrict layer3_bias,
		size2_t layer3_input_size,
		uint_t layer3_in_channels, uint_t layer3_out_channels,
		size2_t layer3_kernel_size, size2_t layer3_stride, size2_t layer3_padding, size2_t layer3_dilation,

		const minibatch_t *__restrict layer4_x, minibatch_t *__restrict layer4_y,
		uint_t layer4_channels, size2_t layer4_input_size,
		size2_t layer4_kernel_size, size2_t layer4_stride, size2_t layer4_padding, size2_t layer4_dilation,

		const minibatch_t *__restrict layer5_x, minibatch_t *__restrict layer5_y, const weight_t *__restrict layer5_weight, const weight_t *__restrict layer5_bias,
		uint_t layer5_in_features, uint_t layer5_out_features,

		const minibatch_t *__restrict layer6_x, minibatch_t *__restrict layer6_y, uint_t layer6_features,

		const minibatch_t *__restrict layer7_x, minibatch_t *__restrict layer7_y, const weight_t *__restrict layer7_weight, const weight_t *__restrict layer7_bias,
		uint_t layer7_in_features, uint_t layer7_out_features
		) {

#define INTERFACE_M_AXI(port_value, bundle_value, depth_value) \
		DO_PRAGMA(HLS INTERFACE m_axi port=port_value offset=slave bundle=bundle_value depth=depth_value)
#define INTERFACE_S_AXILITE(port_value) \
		DO_PRAGMA(HLS INTERFACE s_axilite port=port_value)
#define INTERFACE_CONV(n, b, dx, dy, dw, db) INTERFACE_CONV_(n, b, dx, dy, dw, db)
#define INTERFACE_CONV_(n, b, dx, dy, dw, db) \
		INTERFACE_M_AXI(n##_x, b##_data, dx) \
		INTERFACE_M_AXI(n##_y, b##_data, dy) \
		INTERFACE_M_AXI(n##_weight, b##_param, dw) \
		INTERFACE_M_AXI(n##_bias, b##_param, db) \
		INTERFACE_S_AXILITE(n##_input_size) \
		INTERFACE_S_AXILITE(n##_in_channels) \
		INTERFACE_S_AXILITE(n##_out_channels) \
		INTERFACE_S_AXILITE(n##_kernel_size) \
		INTERFACE_S_AXILITE(n##_stride) \
		INTERFACE_S_AXILITE(n##_padding) \
		INTERFACE_S_AXILITE(n##_dilation)
#define INTERFACE_POOL(n, b, dx, dy) INTERFACE_POOL_(n, b, dx, dy)
#define INTERFACE_POOL_(n, b, dx, dy) \
		INTERFACE_M_AXI(n##_x, b##_data, dx) \
		INTERFACE_M_AXI(n##_y, b##_data, dy) \
		INTERFACE_S_AXILITE(n##_channels) \
		INTERFACE_S_AXILITE(n##_input_size) \
		INTERFACE_S_AXILITE(n##_kernel_size) \
		INTERFACE_S_AXILITE(n##_stride) \
		INTERFACE_S_AXILITE(n##_padding) \
		INTERFACE_S_AXILITE(n##_dilation)
#define INTERFACE_LINE(n, b, dx, dy, dw) INTERFACE_LINE_(n, b, dx, dy, dw, dy)
#define INTERFACE_LINE_(n, b, dx, dy, dw, db) \
		INTERFACE_M_AXI(n##_x, b##_data, dx) \
		INTERFACE_M_AXI(n##_y, b##_data, dy) \
		INTERFACE_M_AXI(n##_weight, b##_param, dw) \
		INTERFACE_M_AXI(n##_bias, b##_param, db) \
		INTERFACE_S_AXILITE(n##_in_features) \
		INTERFACE_S_AXILITE(n##_out_features)
#define INTERFACE_MAPF(n, b, dx) INTERFACE_MAPF_(n, b, dx, dx)
#define INTERFACE_MAPF_(n, b, dx, dy) \
		INTERFACE_M_AXI(n##_x, b##_data, dx) \
		INTERFACE_M_AXI(n##_y, b##_data, dy) \
		INTERFACE_S_AXILITE(n##_features)

#define BUNDLE1 layer1
#define BUNDLE2 layer2
#define BUNDLE5 layer5

#if LAYER6_TANH_CPU
#define BUNDLE6 BUNDLE1
#else
#define BUNDLE6 layer6
#endif

#if REUSE_LAYER_FUNCTIONS
#define BUNDLE3 BUNDLE1
#define BUNDLE4 BUNDLE2
#define BUNDLE7 BUNDLE5
#else
#define BUNDLE3 layer3
#define BUNDLE4 layer4
#define BUNDLE7 layer7
#endif

	INTERFACE_S_AXILITE(return)
	INTERFACE_CONV(layer1, BUNDLE1, 784, 2880, 125, 5)
	INTERFACE_POOL(layer2, BUNDLE2, 2880, 720)
	INTERFACE_CONV(layer3, BUNDLE3, 720, 320, 625, 5)
	INTERFACE_POOL(layer4, BUNDLE4, 320, 80)
	INTERFACE_LINE(layer5, BUNDLE5, 80, 40, 3200)
	INTERFACE_MAPF(layer6, BUNDLE6, 40)
	INTERFACE_LINE(layer7, BUNDLE7, 40, 10, 400)

//#pragma HLS DATAFLOW

	/*
	 * TODO: run these functions in parallel and independently
	 */

	/*
	{
		for (uint_t i = 0; i < 2; i++) {
#pragma HLS UNROLL factor=1
			if (i == 0)
				cnn_Conv2d<0, BATCH_SIZE, value_t, PACK_W>(layer1_x, layer1_y, layer1_weight, layer1_bias, layer1_input_size, layer1_in_channels, layer1_out_channels, layer1_kernel_size, layer1_stride, layer1_padding, layer1_dilation);
			else
				cnn_Conv2d<0, BATCH_SIZE, value_t, PACK_W>(layer3_x, layer3_y, layer3_weight, layer3_bias, layer3_input_size, layer3_in_channels, layer3_out_channels, layer3_kernel_size, layer3_stride, layer3_padding, layer3_dilation);
		}
	}
	*/
	/*

	{
		for (uint_t i = 0; i < 2; i++) {
#pragma HLS UNROLL factor=1
			if (i == 0)
				cnn_MaxPool2d<0, BATCH_SIZE, value_t>(layer2_x, layer2_y, layer2_channels, layer2_input_size, layer2_kernel_size, layer2_stride, layer2_padding, layer2_dilation);
			else
				cnn_MaxPool2d<0, BATCH_SIZE, value_t>(layer4_x, layer4_y, layer4_channels, layer4_input_size, layer4_kernel_size, layer4_stride, layer4_padding, layer4_dilation);
		}
	}
	*/
	/*{
		for (uint_t i = 0; i < 2; i++) {
#pragma HLS UNROLL factor=1
			if (i == 0)
				cnn_Linear<0, BATCH_SIZE, value_t, PACK_W>(layer5_x, layer5_y, layer5_weight, layer5_bias, layer5_in_features, layer5_out_features);
			else
				cnn_Linear<0, BATCH_SIZE, value_t, PACK_W>(layer7_x, layer7_y, layer7_weight, layer7_bias, layer7_in_features, layer7_out_features);
		}
	}*/
//#if !LAYER6_TANH_CPU
	cnn_Tanh<0, BATCH_SIZE, value_t>(layer6_x, layer6_y, layer6_features);
//#endif
}

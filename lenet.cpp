#include "config.h"
#include "cnn_functions.h"
#include "utils.h"

static void lenet1_0(
		const value_t (*__restrict layer1_x)[BATCH_SIZE], value_t (*__restrict layer1_y)[BATCH_SIZE], const value_t *__restrict layer1_weight, const value_t *__restrict layer1_bias,
		size2_t layer1_input_size,
		uint_t layer1_in_channels, uint_t layer1_out_channels,
		size2_t layer1_kernel_size, size2_t layer1_stride, size2_t layer1_padding, size2_t layer1_dilation,

		const value_t (*__restrict layer3_x)[BATCH_SIZE], value_t (*__restrict layer3_y)[BATCH_SIZE], const value_t *__restrict layer3_weight, const value_t *__restrict layer3_bias,
		size2_t layer3_input_size,
		uint_t layer3_in_channels, uint_t layer3_out_channels,
		size2_t layer3_kernel_size, size2_t layer3_stride, size2_t layer3_padding, size2_t layer3_dilation
		) {
#pragma HLS INLINE OFF
	for (uint_t i = 0; i < 2; i++) {
#pragma HLS UNROLL factor=1
		auto x = i == 0 ? layer1_x : layer3_x;
		auto y = i == 0 ? layer1_y : layer3_y;
		auto weight = i == 0 ? layer1_weight : layer3_weight;
		auto bias = i == 0 ? layer1_bias : layer3_bias;
		auto input_size = i == 0 ? layer1_input_size : layer3_input_size;
		auto in_channels = i == 0 ? layer1_in_channels : layer3_in_channels;
		auto out_channels = i == 0 ? layer1_out_channels : layer3_out_channels;
		auto kernel_size = i == 0 ? layer1_kernel_size : layer3_kernel_size;
		auto stride = i == 0 ? layer1_stride : layer3_stride;
		auto padding = i == 0 ? layer1_padding : layer3_padding;
		auto dilation = i == 0 ? layer1_dilation : layer3_dilation;
		cnn_Conv2d<BATCH_SIZE, value_t, PACK_W>(x, y, weight, bias, input_size, in_channels, out_channels, kernel_size, stride, padding, dilation);
	}
}
static void lenet1_1(
		const value_t (*__restrict layer2_x)[BATCH_SIZE], value_t (*__restrict layer2_y)[BATCH_SIZE],
		uint_t layer2_channels, size2_t layer2_input_size,
		size2_t layer2_kernel_size, size2_t layer2_stride, size2_t layer2_padding, size2_t layer2_dilation,

		const value_t (*__restrict layer4_x)[BATCH_SIZE], value_t (*__restrict layer4_y)[BATCH_SIZE],
		uint_t layer4_channels, size2_t layer4_input_size,
		size2_t layer4_kernel_size, size2_t layer4_stride, size2_t layer4_padding, size2_t layer4_dilation
		) {
#pragma HLS INLINE OFF
	for (uint_t i = 0; i < 2; i++) {
#pragma HLS UNROLL factor=1
		auto x = i == 0 ? layer2_x : layer4_x;
		auto y = i == 0 ? layer2_y : layer4_y;
		auto channels = i == 0 ? layer2_channels : layer4_channels;
		auto input_size = i == 0 ? layer2_input_size : layer4_input_size;
		auto kernel_size = i == 0 ? layer2_kernel_size : layer4_kernel_size;
		auto stride = i == 0 ? layer2_stride : layer4_stride;
		auto padding = i == 0 ? layer2_padding : layer4_padding;
		auto dilation = i == 0 ? layer2_dilation : layer4_dilation;
		cnn_MaxPool2d<BATCH_SIZE, value_t>(x, y, channels, input_size, kernel_size, stride, padding, dilation);
	}
}
static void lenet1_2(
		const value_t (*__restrict layer5_x)[BATCH_SIZE], value_t (*__restrict layer5_y)[BATCH_SIZE], const value_t *__restrict layer5_weight, const value_t *__restrict layer5_bias,
		uint_t layer5_in_features, uint_t layer5_out_features,

		const value_t (*__restrict layer7_x)[BATCH_SIZE], value_t (*__restrict layer7_y)[BATCH_SIZE], const value_t *__restrict layer7_weight, const value_t *__restrict layer7_bias,
		uint_t layer7_in_features, uint_t layer7_out_features
		) {
#pragma HLS INLINE OFF
	for (uint_t i = 0; i < 2; i++) {
#pragma HLS UNROLL factor=1
		auto x = i == 0 ? layer5_x : layer7_x;
		auto y = i == 0 ? layer5_y : layer7_y;
		auto weight = i == 0 ? layer5_weight : layer7_weight;
		auto bias = i == 0 ? layer5_bias : layer7_bias;
		auto in_features = i == 0 ? layer5_in_features : layer7_in_features;
		auto out_features = i == 0 ? layer5_out_features : layer7_out_features;
		cnn_Linear<BATCH_SIZE, value_t, PACK_W>(x, y, weight, bias, in_features, out_features);
	}
}
static void lenet1_3(
		const value_t (*__restrict layer6_x)[BATCH_SIZE], value_t (*__restrict layer6_y)[BATCH_SIZE], uint_t layer6_features) {
#pragma HLS INLINE OFF
#if !LAYER6_TANH_CPU
		cnn_Tanh<BATCH_SIZE, value_t>(layer6_x, layer6_y, layer6_features);
#endif
}

void lenet1(
		const value_t (*__restrict layer1_x)[BATCH_SIZE], value_t (*__restrict layer1_y)[BATCH_SIZE], const value_t *__restrict layer1_weight, const value_t *__restrict layer1_bias,
		size2_t layer1_input_size,
		uint_t layer1_in_channels, uint_t layer1_out_channels,
		size2_t layer1_kernel_size, size2_t layer1_stride, size2_t layer1_padding, size2_t layer1_dilation,

		const value_t (*__restrict layer2_x)[BATCH_SIZE], value_t (*__restrict layer2_y)[BATCH_SIZE],
		uint_t layer2_channels, size2_t layer2_input_size,
		size2_t layer2_kernel_size, size2_t layer2_stride, size2_t layer2_padding, size2_t layer2_dilation,

		const value_t (*__restrict layer3_x)[BATCH_SIZE], value_t (*__restrict layer3_y)[BATCH_SIZE], const value_t *__restrict layer3_weight, const value_t *__restrict layer3_bias,
		size2_t layer3_input_size,
		uint_t layer3_in_channels, uint_t layer3_out_channels,
		size2_t layer3_kernel_size, size2_t layer3_stride, size2_t layer3_padding, size2_t layer3_dilation,

		const value_t (*__restrict layer4_x)[BATCH_SIZE], value_t (*__restrict layer4_y)[BATCH_SIZE],
		uint_t layer4_channels, size2_t layer4_input_size,
		size2_t layer4_kernel_size, size2_t layer4_stride, size2_t layer4_padding, size2_t layer4_dilation,

		const value_t (*__restrict layer5_x)[BATCH_SIZE], value_t (*__restrict layer5_y)[BATCH_SIZE], const value_t *__restrict layer5_weight, const value_t *__restrict layer5_bias,
		uint_t layer5_in_features, uint_t layer5_out_features,

		const value_t (*__restrict layer6_x)[BATCH_SIZE], value_t (*__restrict layer6_y)[BATCH_SIZE], uint_t layer6_features,

		const value_t (*__restrict layer7_x)[BATCH_SIZE], value_t (*__restrict layer7_y)[BATCH_SIZE], const value_t *__restrict layer7_weight, const value_t *__restrict layer7_bias,
		uint_t layer7_in_features, uint_t layer7_out_features
		) {

#define INTERFACE_M_AXI(port_value, bundle_value, depth_value) \
		DO_PRAGMA(HLS INTERFACE m_axi port=port_value offset=slave bundle=bundle_value depth=depth_value)
#define INTERFACE_S_AXILITE(port_value) \
		DO_PRAGMA(HLS INTERFACE s_axilite port=port_value)
#define INTERFACE_CONV(n, b, dx, dy, dw, db) INTERFACE_CONV_(n, b, dx, dy, dw, db)
#define INTERFACE_CONV_(n, b, dx, dy, dw, db) \
		INTERFACE_M_AXI(n##_x, b##_data_x, dx) \
		INTERFACE_M_AXI(n##_y, b##_data_y, dy) \
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
#define BUNDLE5 BUNDLE2

#if LAYER6_TANH_CPU
#define BUNDLE6 BUNDLE2
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

	lenet1_0(layer1_x, layer1_y, layer1_weight, layer1_bias,
			layer1_input_size,
			layer1_in_channels, layer1_out_channels,
			layer1_kernel_size, layer1_stride, layer1_padding, layer1_dilation,
			layer3_x, layer3_y, layer3_weight, layer3_bias,
			layer3_input_size,
			layer3_in_channels, layer3_out_channels,
			layer3_kernel_size, layer3_stride, layer3_padding, layer3_dilation);
	lenet1_1(layer2_x, layer2_y,
			layer2_channels, layer2_input_size,
			layer2_kernel_size, layer2_stride, layer2_padding, layer2_dilation,
			layer4_x, layer4_y,
			layer4_channels, layer4_input_size,
			layer4_kernel_size, layer4_stride, layer4_padding, layer4_dilation);
	lenet1_2(layer5_x, layer5_y, layer5_weight, layer5_bias,
			layer5_in_features, layer5_out_features,
			layer7_x, layer7_y, layer7_weight, layer7_bias,
			layer7_in_features, layer7_out_features);
	lenet1_3(layer6_x, layer6_y, layer6_features);
}

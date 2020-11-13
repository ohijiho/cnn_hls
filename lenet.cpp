#include "cnn_functions.h"

#define BATCH_SIZE 16
typedef ap_fixed<32, 8> value_t;
typedef hlslib::DataPack<value_t, 16> minibatch_t;

void lenet1(
		const minibatch_t *layer1_x, minibatch_t *layer1_y, const value_t *layer1_weight, const value_t *layer1_bias,
		size_2_t layer1_input_size,
		size_t layer1_in_channels, size_t layer1_out_channels,
		size_2_t layer1_kernel_size, size_2_t layer1_stride, size_2_t layer1_padding, size_2_t layer1_dilation,

		const minibatch_t *layer2_x, minibatch_t *layer2_y,
		size_t layer2_channels, size_2_t layer2_input_size,
		size_2_t layer2_kernel_size, size_2_t layer2_stride, size_2_t layer2_padding, size_2_t layer2_dilation,

		const minibatch_t *layer3_x, minibatch_t *layer3_y, const value_t *layer3_weight, const value_t *layer3_bias,
		size_2_t layer3_input_size,
		size_t layer3_in_channels, size_t layer3_out_channels,
		size_2_t layer3_kernel_size, size_2_t layer3_stride, size_2_t layer3_padding, size_2_t layer3_dilation,

		const minibatch_t *layer4_x, minibatch_t *layer4_y,
		size_t layer4_channels, size_2_t layer4_input_size,
		size_2_t layer4_kernel_size, size_2_t layer4_stride, size_2_t layer4_padding, size_2_t layer4_dilation,

		const minibatch_t *layer5_x, minibatch_t *layer5_y, const value_t *layer5_weight, const value_t *layer5_bias,
		size_t layer5_in_features, size_t layer5_out_features,

		const minibatch_t *layer6_x, minibatch_t *layer6_y, size_t features,

		const minibatch_t *layer7_x, minibatch_t *layer7_y, const value_t *layer7_weight, const value_t *layer7_bias,
		size_t layer7_in_features, size_t layer7_out_features,
		) {
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS INTERFACE m_axi port=layer1_x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer1_y offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer1_weight offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer1_bias offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=layer1_input_size
#pragma HLS INTERFACE s_axilite port=layer1_in_channels
#pragma HLS INTERFACE s_axilite port=layer1_out_channels
#pragma HLS INTERFACE s_axilite port=layer1_kernel_size
#pragma HLS INTERFACE s_axilite port=layer1_stride
#pragma HLS INTERFACE s_axilite port=layer1_padding
#pragma HLS INTERFACE s_axilite port=layer1_dilation

#pragma HLS INTERFACE m_axi port=layer2_x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer2_y offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=layer2_channels
#pragma HLS INTERFACE s_axilite port=layer2_input_size
#pragma HLS INTERFACE s_axilite port=layer2_kernel_size
#pragma HLS INTERFACE s_axilite port=layer2_stride
#pragma HLS INTERFACE s_axilite port=layer2_padding
#pragma HLS INTERFACE s_axilite port=layer2_dilation

#pragma HLS INTERFACE m_axi port=layer3_x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer3_y offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer3_weight offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer3_bias offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=layer3_input_size
#pragma HLS INTERFACE s_axilite port=layer3_in_channels
#pragma HLS INTERFACE s_axilite port=layer3_out_channels
#pragma HLS INTERFACE s_axilite port=layer3_kernel_size
#pragma HLS INTERFACE s_axilite port=layer3_stride
#pragma HLS INTERFACE s_axilite port=layer3_padding
#pragma HLS INTERFACE s_axilite port=layer3_dilation

#pragma HLS INTERFACE m_axi port=layer4_x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer4_y offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=layer4_channels
#pragma HLS INTERFACE s_axilite port=layer4_input_size
#pragma HLS INTERFACE s_axilite port=layer4_kernel_size
#pragma HLS INTERFACE s_axilite port=layer4_stride
#pragma HLS INTERFACE s_axilite port=layer4_padding
#pragma HLS INTERFACE s_axilite port=layer4_dilation

#pragma HLS INTERFACE m_axi port=layer5_x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer5_y offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer5_weight offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer5_bias offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=layer5_in_features
#pragma HLS INTERFACE s_axilite port=layer5_out_features

#pragma HLS INTERFACE m_axi port=layer6_x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer6_y offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=features

#pragma HLS INTERFACE m_axi port=layer7_x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer7_y offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer7_weight offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=layer7_bias offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=layer7_in_features
#pragma HLS INTERFACE s_axilite port=layer7_out_features

	/*
	 * TODO: run these functions in parallel and independently
	 */

	cnn_Conv2d<BATCH_SIZE, value_t>(layer1_x, layer1_y, layer1_weight, layer1_bias, layer1_input_size, layer1_in_channels, layer1_out_channels, layer1_kernel_size, layer1_stride, layer1_padding, layer1_dilation);
	cnn_MaxPool2d<BATCH_SIZE, value_t>(layer2_x, layer2_y, layer2_channels, layer2_input_size, layer2_kernel_size, layer2_stride, layer2_padding, layer2_dilation);
	cnn_Conv2d<BATCH_SIZE, value_t>(layer3_x, layer3_y, layer3_weight, layer3_bias, layer3_input_size, layer3_in_channels, layer3_out_channels, layer3_kernel_size, layer3_stride, layer3_padding, layer3_dilation);
	cnn_MaxPool2d<BATCH_SIZE, value_t>(layer4_x, layer4_y, layer4_channels, layer4_input_size, layer4_kernel_size, layer4_stride, layer4_padding, layer4_dilation);
	cnn_Linear<BATCH_SIZE, value_t>(layer5_x, layer5_y, layer5_weight, layer5_bias, layer5_in_features, layer5_out_features);
	cnn_Tanh<BATCH_SIZE, value_t>(layer6_x, layer6_y, features);
	cnn_Linear<BATCH_SIZE, value_t>(layer7_x, layer7_y, layer7_weight, layer7_bias, layer7_in_features, layer7_out_features);
}

#include "cnn_functions.h"

void cnn_Conv2d_top(const minibatch_t *x, minibatch_t *y, const value_t *weight, const value_t *bias,
		size_2_t input_size,
		size_t in_channels, size_t out_channels,
		size_2_t kernel_size, size_2_t stride, size_2_t padding, size_2_t dilation) {
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=y offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=input_size
#pragma HLS INTERFACE s_axilite port=in_channels
#pragma HLS INTERFACE s_axilite port=out_channels
#pragma HLS INTERFACE s_axilite port=kernel_size
#pragma HLS INTERFACE s_axilite port=stride
#pragma HLS INTERFACE s_axilite port=padding
#pragma HLS INTERFACE s_axilite port=dilation

//	cnn_Conv2d<BATCH_SIZE, value_t>(x, y, weight, bias, input_size, in_channels, out_channels, kernel_size, stride, padding, dilation);
}

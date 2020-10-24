#ifndef __CNN_HLS_FUNCTIONS_H__
#define __CNN_HLS_FUNCTIONS_H__
#include "types.h"
#include "matmul.h"
#include <math.h>
#include <hlslib/xilinx/Operators.h>
#include <hlslib/xilinx/DataPack.h>
template<typename T, typename RAM_A, typename RAM_B, typename RAM_C>
void matmul(RAM_A a, RAM_B b, RAM_C c,
		size_t size_m, size_t size_k, size_t size_n) {
	matmul_dynamic_ram_naive<T, hlslib::op::Product, hlslib::op::Sum>(
			a, b, c, size_m, size_k, size_n);
}

template<size_t batch_size, typename T,
		typename RAM_x, typename RAM_y, typename RAM_weight, typename RAM_bias>
void cnn_Conv2d(RAM_x x, RAM_y y, RAM_weight weight, RAM_bias bias,
		size_2_t input_size,
		size_t in_channels, size_t out_channels,
		size_2_t kernel_size, size_2_t stride, size_2_t padding, size_2_t dilation) {
	/*
	 * T x[in_channels][input_size][batch_size];
	 * T y[out_channels][output_size][batch_size];
	 * T weight[in_channels][kernel_size][out_channels];
	 */

#ifdef ______PSEUDO_CODE_______
	for (ptrdiff_t i = 0; i < (ptrdiff_t)in_channels; i++) {
		a << weight;
		b << im2col(x[i]);
	}
	matmul_stream_n_bias_transpose_a(a, b, c, bias,
			out_channels, in_channels, batch_size * output_size);
	c >> y;
#endif
}

template<size_t batch_size, typename T, typename RAM_x, typename RAM_y>
void cnn_MaxPool2d(RAM_x x, RAM_y y,
		size_t channels, size_2_t input_size,
		size_2_t kernel_size, size_2_t stride, size_2_t padding, size_2_t dilation) {
	using pack_t = hlslib::DataPack<T, batch_size>;
	/*
	 * pack_t x[channels][input_size];
	 * pack_t y[channels][output_size][batch_size];
	 */
	const size_2_t output_size = {/* FIXME */};
	for (ptrdiff_t ic = 0; ic < (ptrdiff_t)channels; ic++) {
		for (ptrdiff_t ih = 0; ih < (ptrdiff_t)output_size.height; ih++) {
			for (ptrdiff_t iw = 0; iw < (ptrdiff_t)output_size.width; iw++) {
				pack_t m = -INFINITY;
				for (ptrdiff_t kh = 0; kh < (ptrdiff_t)kernel_size.height; kh++) {
					for (ptrdiff_t kw = 0; kw < (ptrdiff_t)kernel_size.width; kw++) {
						pack_t v = x[ic * input_size.height * input_size.width +
									 0 /* FIXME */];
						for (ptrdiff_t ib = 0; ib < (ptrdiff_t)batch_size; ib++) {
#pragma HLS UNROLL
							if (v[ib] > m[ib])
								m[ib] = v[ib];
						}
					}
				}
				y[ic * output_size.height * output_size.width +
				  ih * output_size.width + iw] = m;
			}
		}
	}

}

template<size_t batch_size, typename T,
		typename RAM_x, typename RAM_y, typename RAM_weight, typename RAM_bias>
void cnn_Linear(RAM_x x, RAM_y y, RAM_weight weight, RAM_bias bias,
		size_t in_features, size_t out_features) {
	/*
	 * T x[in_features][batch_size];
	 * T y[out_features][batch_size];
	 * T weight[in_features][out_features];
	 * T bias[out_features];
	 */
	matmul_n_bias_transpose_a<T>(weight, x, y, bias, out_features, in_features, batch_size);
}

template<class Function, typename RAM_x, typename RAM_y>
void map_function(RAM_x x, RAM_y y, size_t size) {
	for (ptrdiff_t i = 0; i < size; i++) {
		y[i] = Function::Apply(x[i]);
	}
}


template<size_t batch_size, typename T, typename RAM_x, typename RAM_y>
void cnn_ReLU(RAM_x x, RAM_y y, size_t features) {
	/*
	 * T x[features][batch_size];
	 * T y[features][batch_size];
	 */
	struct Function {
		static T Apply(T &&x) {
#pragma HLS INLINE
			return x < 0 ? 0 : x;
		}
	};
	map_function<Function>(x, y, batch_size * features);
}

template<size_t batch_size, typename T, typename RAM_x, typename RAM_y>
void cnn_Tanh(RAM_x x, RAM_y y, size_t features) {
	/*
	 * T x[features][batch_size];
	 * T y[features][batch_size];
	 */
	struct Function {
		static T Apply(T &&x) {
#pragma HLS INLINE
			return tanh(x);
		}
	};
	map_function<Function>(x, y, batch_size * features);
}

#endif

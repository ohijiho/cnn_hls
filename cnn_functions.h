#ifndef __CNN_HLS_FUNCTIONS_H__
#define __CNN_HLS_FUNCTIONS_H__
#include "types.h"
#include "matmul.h"
//#include <math.h>
#include <hls_math.h>
#include "im2col.h"
#include "utils.h"
#include <hlslib/xilinx/Operators.h>

template<typename T, typename RAM_A, typename RAM_B, typename RAM_C>
void matmul(RAM_A a, RAM_B b, RAM_C c,
		size_t size_m, size_t size_k, size_t size_n) {
	matmul_ram_naive<T>(
			a, b, c, size_m, size_k, size_n);
}

template<size_t batch_size, typename T,
		typename RAM_x, typename RAM_y, typename RAM_weight, typename RAM_bias>
void cnn_Conv2d(RAM_x x, RAM_y y, RAM_weight weight, RAM_bias bias,
		size_2_t input_size,
		size_t in_channels, size_t out_channels,
		size_2_t kernel_size, size_2_t stride, size_2_t padding, size_2_t dilation) {
	using pack_t = hlslib::DataPack<T, batch_size>;
	/*
	 * pack_t x[in_channels][input_size];
	 * pack_t y[out_channels][output_size];
	 * T weight[in_channels][kernel_size][out_channels];
	 */

	using im2col_t = ram_im2col<batch_size, 1, T, RAM_x>;
	im2col_t im2col(x, input_size, in_channels, out_channels, kernel_size, stride, padding, dilation);
	matmul_m_bias_transpose_a<T, 1, batch_size>(weight, im2col, y, bias,
			im2col.size_m, im2col.size_k, im2col.size_n);
}

template<size_t batch_size, typename T, typename RAM_x, typename RAM_y>
void cnn_MaxPool2d(RAM_x x, RAM_y y,
		size_t channels, size_2_t input_size,
		size_2_t kernel_size, size_2_t stride, size_2_t padding, size_2_t dilation) {
	using pack_t = hlslib::DataPack<T, batch_size>;
	/*
	 * pack_t x[channels][input_size];
	 * pack_t y[channels][output_size];
	 */

#if 1
	using im2col_t = ram_im2col<batch_size, 1, T, RAM_x>;
//	std::cout << "===min: " << MaxIdentity<T>::value() << std::endl;
	im2col_t im2col(x, input_size, channels, 1, kernel_size, stride, padding, dilation, MaxIdentity<T>::value());
	for (ptrdiff_t ci = 0; ci < (ptrdiff_t)channels; ci++) {
		for (ptrdiff_t oi = 0; oi < (ptrdiff_t)im2col.size_n; oi++) {
			pack_t m = MaxIdentity<T>::value();
			for (ptrdiff_t ki = 0; ki < (ptrdiff_t)kernel_size.area(); ki++) {
				const pack_t v = im2col.get(ci, ki, oi);
				for (ptrdiff_t ib = 0; ib < (ptrdiff_t)batch_size; ib++) {
#pragma HLS UNROLL
					m[ib] = std::max((T)m[ib], v[ib]);
				}
			}
			y[ci * im2col.size_n + oi] = m;
		}
	}
#else
	const size_2_t output_size = calc_output_size(input_size, kernel_size, stride, padding, dilation);
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
							m[ib] = std::max(m[ib], v[ib]);
						}
					}
				}
				y[ic * output_size.height * output_size.width +
				  ih * output_size.width + iw] = m;
			}
		}
	}
#endif

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
	matmul_m_bias_transpose_a<T, 1, batch_size>(weight, x, y, bias, out_features, in_features, 1);
}

template<size_t batch_size, typename T,
		class Function, typename RAM_x, typename RAM_y>
void map_function(RAM_x x, RAM_y y, size_t size) {
#pragma HLS INLINE
	using pack_t = hlslib::DataPack<T, batch_size>;
	for (ptrdiff_t i = 0; i < (ptrdiff_t)size; i++) {
		const pack_t tx = x[i];
		pack_t ty;
		for (ptrdiff_t j = 0; j < (ptrdiff_t)batch_size; j++) {
#pragma HLS UNROLL
			ty[j] = Function::Apply(tx[j]);
		}
		y[i] = ty;
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
	map_function<batch_size, T, Function>(x, y, features);
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
			T y = hls::tanh(x);
			/*
			 * TODO: BUG?
			 * 		hls::tanh drops sign
			 */
			if (x < 0 && y > 0)
				y = -y;
			return y;
		}
	};
	map_function<batch_size, T, Function>(x, y, features);
}

#endif

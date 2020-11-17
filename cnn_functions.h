#ifndef __CNN_HLS_FUNCTIONS_H__
#define __CNN_HLS_FUNCTIONS_H__
#include "types.h"
#include "matmul.h"
//#include <math.h>
#include <hls_math.h>
#include "im2col.h"
#include "utils.h"
#include <hlslib/xilinx/Operators.h>
#include "operators.h"

template<typename T, typename RAM_A, typename RAM_B, typename RAM_C>
void matmul(RAM_A a, RAM_B b, RAM_C c,
		uint_t size_m, uint_t size_k, uint_t size_n) {
	matmul_ram_naive<T>(
			a, b, c, size_m, size_k, size_n);
}

template<uint_t dup_func, uint_t batch_size, typename T, uint_t pack_w = 1,
		typename RAM_x, typename RAM_y, typename RAM_weight, typename RAM_bias>
void cnn_Conv2d(RAM_x x, RAM_y y, RAM_weight weight, RAM_bias bias,
		size2_t input_size,
		uint_t in_channels, uint_t out_channels,
		size2_t kernel_size, size2_t stride, size2_t padding, size2_t dilation) {
	/*
	 * pack_t x[in_channels][input_size];
	 * pack_t y[out_channels][output_size];
	 * T weight[in_channels][kernel_size][out_channels];
	 */
	using row_t = hlslib::DataPack<T, batch_size>;
	using col_t = hlslib::DataPack<T, pack_w>;
#define WHICH 1
#if WHICH == 0
	using im2col_t = ram_im2col<batch_size, T, RAM_x>;
	im2col_t im2col(x, input_size, in_channels, out_channels, kernel_size, stride, padding, dilation);
	matmul_m_bias_transpose_a<dup_func, T, pack_w, pack_w, batch_size, batch_size>(weight, im2col, y, bias,
			im2col.size_m, im2col.size_k, im2col.size_n);
#elif WHICH == 1
	using im2col_t = ram_im2col<batch_size, T, RAM_x>;
	im2col_t im2col(x, input_size, in_channels, out_channels, kernel_size, stride, padding, dilation);
	row_t b_bram[std::max(1 * 5 * 5 * 24 * 24, 5 * 5 * 5 * 8 * 8)];
	row_t c_bram[std::max(5 * 24 * 24, 5 * 8 * 8)];
	im2col.dump_all(b_bram);
	matmul_m_bias_transpose_a<dup_func, T, pack_w, pack_w, batch_size, batch_size>(weight, b_bram, c_bram, bias,
			im2col.size_m, im2col.size_k, im2col.size_n);
	for (uint_t i = 0; i < im2col.size_m * im2col.size_n; i++) {
		y[i] = c_bram[i];
	}
#elif WHICH == 2
	using im2col_t = iter_im2col<pack_w, batch_size, T, RAM_x>;
	const uint_t block_m = (5 - 1) / pack_w + 1, block_k = 5, block_n = 4;
	im2col_t im2col(x, input_size, in_channels, out_channels, kernel_size, stride, padding, dilation, 4, 5);
	col_t a_bram[block_m * block_k];
	row_t b_bram[block_k * block_n];
	row_t c_bram[block_m * block_n];
	for (uint_t i = 0; i < im2col.size_m; i += block_m) {
		for (im2col.reset(); im2col.dump(b_bram, true);) {
			const auto &res = im2col.last_result;
			load_bias<0, T, pack_w, batch_size>(c_bram, bias, im2col.size_m, im2col.size_n, i, block_m, block_n);
			matmul_acc_transpose_a<dup_func, T, pack_w, pack_w, batch_size, batch_size>(
					weight, b_bram, c_bram, block_m, block_k, block_n);
		}
	}
#else
#error No implementation
#endif
#undef WHICH
}

template<uint_t dup_func, uint_t batch_size, typename T, typename RAM_x, typename RAM_y>
void cnn_AvgPool2d(RAM_x x, RAM_y y,
		uint_t channels, size2_t input_size,
		size2_t kernel_size, size2_t stride, size2_t padding, size2_t dilation) {
	/*
	 * pack_t x[channels][input_size];
	 * pack_t y[channels][output_size];
	 */

	using im2col_t = ram_im2col<batch_size, T, RAM_x>;
	im2col_t im2col(x, input_size, channels, 1, kernel_size, stride, padding, dilation, MaxIdentity<T>::value());
	map_reduce<dup_func, T, batch_size, 1>((T)1 / (T)kernel_size.area(), im2col, y, 0,
			channels, kernel_size.area(), im2col.size_n);
}

template<uint_t dup_func, uint_t batch_size, typename T, typename RAM_x, typename RAM_y>
void cnn_MaxPool2d(RAM_x x, RAM_y y,
		uint_t channels, size2_t input_size,
		size2_t kernel_size, size2_t stride, size2_t padding, size2_t dilation) {
	/*
	 * pack_t x[channels][input_size];
	 * pack_t y[channels][output_size];
	 */
#if 1
	using im2col_t = ram_im2col<batch_size, T, RAM_x>;
	im2col_t im2col(x, input_size, channels, 1, kernel_size, stride, padding, dilation, MaxIdentity<T>::value());
	map_reduce<dup_func, T, batch_size, 1, RightOp<T>, hlslib::op::Max<T>>(0, im2col, y, MaxIdentity<T>::value(),
			channels, kernel_size.area(), im2col.size_n);
#elif 1
	using im2col_t = ram_im2col<batch_size, 1, T, RAM_x>;
//	std::cout << "===min: " << MaxIdentity<T>::value() << std::endl;
	im2col_t im2col(x, input_size, channels, 1, kernel_size, stride, padding, dilation, MaxIdentity<T>::value());
	for (uint_t ci = 0; ci < channels; ci++) {
		for (uint_t oi = 0; oi < im2col.size_n; oi++) {
			row_t m = MaxIdentity<T>::value();
			for (uint_t ki = 0; ki < kernel_size.area(); ki++) {
				const row_t v = im2col.get(ci, ki, oi);
				for (uint_t ib = 0; ib < batch_size; ib++) {
#pragma HLS UNROLL
//					m[ib] = std::max((T)m[ib], v[ib]);
					/*
					 * remove conditional statement
					 */
					int mux_select = (int)(v[ib] > (T)m[ib]);
					const T mux_inputs[2] = {m[ib], v[ib]};
					m[ib] = mux_inputs[mux_select];
				}
			}
			y[ci * im2col.size_n + oi] = m;
		}
	}
#else
	const size2_t output_size = calc_output_size(input_size, kernel_size, stride, padding, dilation);
	for (uint_t ic = 0; ic < channels; ic++) {
		for (uint_t ih = 0; ih < output_size.height; ih++) {
			for (uint_t iw = 0; iw < output_size.width; iw++) {
				row_t m = -INFINITY;
				for (uint_t kh = 0; kh < kernel_size.height; kh++) {
					for (uint_t kw = 0; kw < kernel_size.width; kw++) {
						row_t v = x[ic * input_size.height * input_size.width +
									 0 /* FIXME */];
						for (uint_t ib = 0; ib < batch_size; ib++) {
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

template<uint_t dup_func, uint_t batch_size, typename T, uint_t pack_w = 1,
		typename RAM_x, typename RAM_y, typename RAM_weight, typename RAM_bias>
void cnn_Linear(RAM_x x, RAM_y y, RAM_weight weight, RAM_bias bias,
		uint_t in_features, uint_t out_features) {
	/*
	 * T x[in_features][batch_size];
	 * T y[out_features][batch_size];
	 * T weight[in_features][out_features];
	 * T bias[out_features];
	 */
	using pack_t = hlslib::DataPack<T, batch_size>;
	/*
	 * pack_t x[in_channels][input_size];
	 * pack_t y[out_channels][output_size];
	 * T weight[in_channels][kernel_size][out_channels];
	 */

	matmul_m_bias_transpose_a<dup_func, T, pack_w, pack_w, batch_size, batch_size>(weight, x, y, bias, out_features, in_features, 1);
//	cnn_Conv2d<dup_func, batch_size, T>(x, y, weight, bias,
//			{1, 1}, in_features, out_features, {1, 1}, {1, 1}, {0, 0}, {1, 1});

//	using im2col_t = ram_im2col<batch_size, 1, T, RAM_x>;
//	im2col_t im2col(x, {1, 1}, in_features, out_features, {1, 1}, {1, 1}, {0, 0}, {1, 1});
//	matmul_m_bias_transpose_a<dup_func, T, 1, 1, batch_size, batch_size>(weight, im2col, y, bias,
//			out_features, in_features, 1);

//	struct ram_t {
//		pack_t operator[](int_t i) {
//			return (T)0;
//		}
//	} ram;
//	matmul_m_bias_transpose_a<dup_func, T, 1, 1, batch_size, batch_size>(weight, ram, y, bias, out_features, in_features, 1);
}

template<uint_t dup_func, uint_t batch_size, typename T, uint_t unroll_factor,
		class Function, typename RAM_x, typename RAM_y>
void map_function(RAM_x x, RAM_y y, uint_t size) {
#pragma HLS INLINE
	using pack_t = hlslib::DataPack<T, batch_size>;
	for (uint_t i = 0; i < size; i++) {
		const pack_t tx = x[i];
		pack_t ty;
		for (uint_t j = 0; j < batch_size; j++) {
#pragma HLS UNROLL factor=unroll_factor
			ty[j] = Function::Apply(tx[j]);
		}
		y[i] = ty;
	}
}


template<uint_t dup_func, uint_t batch_size, typename T, uint_t unroll_factor = 1, typename RAM_x, typename RAM_y>
void cnn_ReLU(RAM_x x, RAM_y y, uint_t features) {
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
	map_function<dup_func, batch_size, T, unroll_factor, Function>(x, y, features);
}

template<uint_t dup_func, uint_t batch_size, typename T, uint_t unroll_factor = 1, typename RAM_x, typename RAM_y>
void cnn_Tanh(RAM_x x, RAM_y y, uint_t features) {
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
	map_function<dup_func, batch_size, T, unroll_factor, Function>(x, y, features);
}

#endif

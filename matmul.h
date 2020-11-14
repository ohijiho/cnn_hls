#ifndef __CNN_HLS_MATMUL_H__
#define __CNN_HLS_MATMUL_H__
#include "types.h"
#include <hlslib/xilinx/Operators.h>
#include <sstream>
#include <hlslib/xilinx/DataPack.h>

template<typename T, typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>,
	typename RAM_a, typename RAM_b, typename RAM_c>
void matmul_ram_naive(RAM_a a, RAM_b b, RAM_c c,
		size_t size_m, size_t size_k, size_t size_n) {
	for (ptrdiff_t i = 0; i < (ptrdiff_t)size_m; i++) {
		for (ptrdiff_t j = 0; j < (ptrdiff_t)size_n; j++) {
			T acc = OperatorReduce::identity();
			for (ptrdiff_t k = 0; k < (ptrdiff_t)size_k; k++) {
				acc = OperatorReduce::Apply(acc, OperatorMap::Apply(
						a[i * size_k + k], b[k * size_m + j]));
			}
			c[i * size_n + j] = acc;
		}
	}
}

template<typename T, size_t pack_m = 1, size_t pack_n = 1,
		typename RAM_a, typename RAM_b, typename RAM_c, typename RAM_bias,
		typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>>
void matmul_m_bias_transpose_a(RAM_a a, RAM_b b, RAM_c c, RAM_bias bias,
		size_t size_m, size_t size_k, size_t size_n) {
	typedef hlslib::DataPack<T, pack_n> row_t;
	typedef hlslib::DataPack<T, pack_m> col_t;
	typedef hlslib::DataPack<row_t, pack_m> mat_t;
	for (ptrdiff_t i = 0; i < (ptrdiff_t)size_m; i++) {
		for (ptrdiff_t j = 0; j < (ptrdiff_t)size_n; j++) {
			row_t acc[pack_m];
			{
				const col_t biasbuf = bias[i];
				for (ptrdiff_t ki = 0; ki < (ptrdiff_t)pack_m; ki++) {
#pragma HLS UNROLL
					acc[ki].Fill(biasbuf[ki]);
				}
			}
			for (ptrdiff_t k = 0; k < (ptrdiff_t)size_k; k++) {
				const col_t abuf = a[k * size_m + i];
				const row_t bbuf = b[k * size_n + j];
				for (ptrdiff_t ki = 0; ki < (ptrdiff_t)pack_m; ki++) {
#pragma HLS UNROLL
					for (ptrdiff_t kj = 0; kj < (ptrdiff_t)pack_n; kj++) {
#pragma HLS UNROLL
						acc[ki][kj] = OperatorReduce::Apply((T)acc[ki][kj], OperatorMap::Apply(
								abuf[ki], bbuf[kj]));
					}
				}
			}
			for (ptrdiff_t ki = 0; ki < (ptrdiff_t)pack_m; ki++) {
#pragma HLS UNROLL
				c[(i * pack_m + ki) * size_n + j] = acc[ki];
			}
		}
	}
}

#endif

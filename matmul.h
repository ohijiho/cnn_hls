#ifndef __CNN_HLS_MATMUL_H__
#define __CNN_HLS_MATMUL_H__
#include "types.h"
#include <hlslib/xilinx/Operators.h>
#include <sstream>
#include <hlslib/xilinx/DataPack.h>

template<typename T, typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>,
	typename RAM_a, typename RAM_b, typename RAM_c>
void matmul_ram_naive(RAM_a a, RAM_b b, RAM_c c,
		uint_t size_m, uint_t size_k, uint_t size_n) {
	for (uint_t i = 0; i < size_m; i++) {
		for (uint_t j = 0; j < size_n; j++) {
			T acc = OperatorReduce::identity();
			for (uint_t k = 0; k < size_k; k++) {
				acc = OperatorReduce::Apply(acc, OperatorMap::Apply(
						a[i * size_k + k], b[k * size_m + j]));
			}
			c[i * size_n + j] = acc;
		}
	}
}

//template<uint_t dup_func, typename T, uint_t pack_m = 1, uint_t unroll_m = 1, uint_t pack_n = 1, uint_t unroll_n = 1,
//		typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>,
//		typename RAM_a, typename RAM_b, typename RAM_c, typename RAM_bias>
//void matmul_row_bias_transpose_a(RAM_a a, RAM_b b, RAM_c c, RAM_bias bias,
//		uint_t size_m, uint_t size_k, uint_t size_n) {
//	using col_t = hlslib::DataPack<T, pack_m>;
//	using row_t = hlslib::DataPack<T, pack_n>;
//	for (uint_t i = 0; i < size_m; i++) {
//		for (uint_t j = 0; j < size_n; j++) {
//			row_t acc[pack_m];
//			{
//				const col_t biasbuf = bias[i];
//				for (uint_t ki = 0; ki < pack_m; ki++) {
//#pragma HLS UNROLL factor=unroll_m
//					for (uint_t kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//						acc[ki][kj] = biasbuf[ki];
//					}
//				}
//			}
//			for (uint_t k = 0; k < size_k; k++) {
//				const col_t abuf = a[k * size_m + i];
//				const row_t bbuf = b[k * size_n + j];
//				for (uint_t ki = 0; ki < pack_m; ki++) {
//#pragma HLS UNROLL factor=unroll_m
//					for (uint_t kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//						acc[ki][kj] = OperatorReduce::Apply((T)acc[ki][kj], OperatorMap::Apply(
//								abuf[ki], bbuf[kj]));
//					}
//				}
//			}
//			for (uint_t ki = 0; ki < pack_m; ki++) {
//#pragma HLS UNROLL factor=unroll_m
//				c[(i * pack_m + ki) * size_n + j] = acc[ki];
//			}
//		}
//	}
//}

template<typename T, uint_t pack_m = 1, uint_t unroll_m = 1, uint_t pack_n = 1, uint_t unroll_n = 1,
		typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>,
		typename RAM_a, typename RAM_b, typename RAM_c, typename RAM_bias>
void matmul_row_bias_transpose_a(RAM_a a, RAM_b b, RAM_c c, RAM_bias bias,
		uint_t size_m, uint_t size_k, uint_t size_n) {
	using col_t = hlslib::DataPack<T, pack_m>;
	using row_t = hlslib::DataPack<T, pack_n>;
	const uint_t num_m_pack = (size_m - 1) / pack_m + 1;
	const uint_t pack_m_size_n = pack_m * size_n;
	for (uint_t i = 0, cri = 0, off_c_row = 0; cri < size_m; i++, cri += pack_m, off_c_row += pack_m_size_n) {
		for (uint_t j = 0; j < size_n; j++) {
			row_t acc[pack_m];
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=unroll_m
			T acc_part[pack_m][pack_n];
#pragma HLS ARRAY_PARTITION variable=acc_part cyclic factor=unroll_m dim=1
#pragma HLS ARRAY_PARTITION variable=acc_part cyclic factor=unroll_n dim=2
			{
				const col_t biasbuf = bias[i];
				T biasbuf_part[pack_m];
#pragma HLS ARRAY_PARTITION variable=biasbuf_part cyclic factor=unroll_m
				biasbuf >> biasbuf_part;
				for (uint_t ki = 0; ki < pack_m; ki++) {
#pragma HLS UNROLL factor=unroll_m
					for (uint_t kj = 0; kj < pack_n; kj++) {
#pragma HLS UNROLL factor=unroll_n
						acc_part[ki][kj] = biasbuf_part[ki];
					}
				}
			}
			for (uint_t k = 0, aoff = i, boff = j; k < size_k; k++, aoff += num_m_pack, boff += size_n) {
				const col_t abuf = a[aoff];
				const row_t bbuf = b[boff];
				T abuf_part[pack_m], bbuf_part[pack_n];
#pragma HLS ARRAY_PARTITION variable=abuf_part cyclic factor=unroll_m
#pragma HLS ARRAY_PARTITION variable=bbuf_part cyclic factor=unroll_n
				abuf >> abuf_part;
				bbuf >> bbuf_part;
				for (uint_t ki = 0; ki < pack_m; ki++) {
#pragma HLS UNROLL factor=unroll_m
					for (uint_t kj = 0; kj < pack_n; kj++) {
#pragma HLS UNROLL factor=unroll_n
						acc_part[ki][kj] = OperatorReduce::Apply(acc_part[ki][kj], OperatorMap::Apply(
								abuf_part[ki], bbuf_part[kj]));
					}
				}
			}
			for (uint_t ki = 0; ki < pack_m; ki++) {
#pragma HLS UNROLL factor=unroll_m
				acc[ki] << acc_part[ki];
			}
			for (uint_t ki = 0, coff = off_c_row + j; ki < pack_m; ki++, coff += size_n) {
#pragma HLS UNROLL factor=unroll_m
				if (cri + ki < size_m) {
					c[coff] = acc[ki];
				}
			}
		}
	}
}

//template<uint_t dup_func, typename T, uint_t pack_m = 1, uint_t unroll_m = 1, uint_t pack_n = 1, uint_t unroll_n = 1,
//		typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>,
//		typename RAM_a, typename RAM_b, typename RAM_c, typename RAM_bias>
//void matmul_row_bias_transpose_a(RAM_a a, RAM_b b, RAM_c c, RAM_bias bias,
//		uint_t size_m, uint_t size_k, uint_t size_n) {
//	using col_t = hlslib::DataPack<T, 1>;
//	using row_t = hlslib::DataPack<T, pack_n>;
//	for (uint_t i = 0; i < size_m; i++) {
//		for (uint_t j = 0; j < size_n; j++) {
//			row_t acc[1];
//			T acc_part[1][pack_n];
//#pragma HLS ARRAY_PARTITION variable=acc_part cyclic factor=unroll_n dim=1
//			{
//				const col_t biasbuf = bias[i];
//				T biasbuf_part[1];
//				biasbuf >> biasbuf_part;
//				for (uint_t kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//					acc_part[0][kj] = biasbuf_part[0];
//				}
//			}
//			for (uint_t k = 0; k < size_k; k++) {
//				const col_t abuf = a[k * size_m + i];
//				const row_t bbuf = b[k * size_n + j];
//				T abuf_part[1], bbuf_part[pack_n];
//#pragma HLS ARRAY_PARTITION variable=bbuf_part cyclic factor=unroll_n
//				abuf >> abuf_part;
//				bbuf >> bbuf_part;
//				for (uint_t kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//					acc_part[0][kj] = OperatorReduce::Apply(acc_part[0][kj], OperatorMap::Apply(
//							abuf_part[0], bbuf_part[kj]));
//				}
//			}
//			acc[0] << acc_part[0];
//			c[i * size_n + j] = acc[0];
//		}
//	}
//}

//template<uint_t dup_func, typename T, uint_t pack_n = 1, uint_t unroll_n = 1,
//		typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>,
//		typename RAM_b, typename RAM_c>
//void map_reduce(T a, RAM_b b, RAM_c c, T bias,
//		uint_t size_m, uint_t size_k, uint_t size_n) {
//	typedef hlslib::DataPack<T, pack_n> row_t;
//	for (uint_t i = 0; i < size_m; i++) {
//		for (uint_t j = 0; j < size_n; j++) {
//			row_t acc = bias;
//			for (uint_t k = 0; k < size_k; k++) {
//				const row_t bbuf = b[(i * size_k + k) * size_n + j];
//				for (uint_t kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//					acc[kj] = OperatorReduce::Apply((T)acc[kj], bbuf[kj]);
//				}
//			}
//			for (uint_t kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//				acc[kj] = OperatorMap::Apply(a, acc[kj]);
//			}
//			c[i * size_n + j] = acc;
//		}
//	}
//}

template<typename T, uint_t pack_n = 1, uint_t unroll_n = 1,
		typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>,
		typename RAM_b, typename RAM_c>
void map_reduce(T a, RAM_b b, RAM_c c, T bias,
		uint_t size_m, uint_t size_k, uint_t size_n) {
	using row_t = hlslib::DataPack<T, pack_n>;
	using row_part_t = hlslib::DataPack<T, unroll_n>;

	const uint_t b_ch_size = size_k * size_n;
	for (uint_t i = 0, coff = 0, off_b_i = 0; i < size_m; i++, off_b_i += b_ch_size) {
		for (uint_t j = 0; j < size_n; j++, coff++) {
			row_t acc = bias;
			T acc_part[pack_n];
#pragma HLS ARRAY_PARTITION variable=acc_part cyclic factor=unroll_n
			acc >> acc_part;
			for (uint_t k = 0, boff = off_b_i + j; k < size_k; k++, boff += size_n) {
				const row_t bbuf = b[boff];
				T bbuf_part[pack_n];
#pragma HLS ARRAY_PARTITION variable=bbuf_part cyclic factor=unroll_n
				bbuf >> bbuf_part;
				for (uint_t kj = 0; kj < pack_n; kj++) {
#pragma HLS UNROLL factor=unroll_n
					acc_part[kj] = OperatorReduce::Apply(acc_part[kj], bbuf_part[kj]);
				}
			}
			for (uint_t kj = 0; kj < pack_n; kj++) {
#pragma HLS UNROLL factor=unroll_n
				acc_part[kj] = OperatorMap::Apply(a, acc_part[kj]);
			}
			acc << acc_part;
			c[coff] = acc;
		}
	}
}

//template<uint_t dup_func, typename T, uint_t pack_m = 1, uint_t unroll_m = 1, uint_t pack_n = 1, uint_t unroll_n = 1,
//		typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>,
//		typename RAM_a, typename RAM_b, typename RAM_c, typename RAM_bias>
//void matmul_row_bias_transpose_a(RAM_a a, RAM_b b, RAM_c c, RAM_bias bias,
//		uint_t size_m, uint_t size_k, uint_t size_n) {
//	for (uint_t i = 0; i < size_m; i++) {
//		for (uint_t j = 0; j < size_n; j++) {
//			T acc[pack_m][pack_n];
//#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=unroll_n dim=2
//#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=unroll_m dim=1
//			{
//				T biasbuf[pack_m];
//#pragma HLS ARRAY_PARTITION variable=biasbuf cyclic factor=unroll_m dim=1
//				for (uint_t ki = 0; ki < pack_m; ki++) {
//#pragma HLS UNROLL factor=unroll_m
//					biasbuf[ki] = bias[i][ki];
//				}
//				for (uint_t ki = 0; ki < pack_m; ki++) {
//#pragma HLS UNROLL factor=unroll_m
//					for (uint_t kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//						acc[ki][kj] = biasbuf[ki];
//					}
//				}
////				for (uint_t ki = 0; ki < pack_m; ki++) {
////#pragma HLS UNROLL factor=unroll_m
////					for (uint_t kj = 0; kj < pack_n; kj++) {
////#pragma HLS UNROLL factor=unroll_n
////						acc[ki][kj] = bias[i][ki];
////					}
////				}
//			}
//			for (uint_t k = 0; k < size_k; k++) {
//				T abuf[pack_m];
//#pragma HLS ARRAY_PARTITION variable=abuf cyclic factor=unroll_m dim=1
//				T bbuf[pack_n];
//#pragma HLS ARRAY_PARTITION variable=bbuf cyclic factor=unroll_n dim=1
//				for (uint_t ai = k * size_m + i, ki = 0; ki < pack_m; ki++) {
//#pragma HLS UNROLL factor=unroll_m
//					abuf[ki] = a[ai][ki];
//				}
//				for (uint_t bi = k * size_n + j, kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//					bbuf[kj] = b[bi][kj];
//				}
//				for (uint_t ki = 0; ki < pack_m; ki++) {
//#pragma HLS UNROLL factor=unroll_m
//					for (uint_t kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//						acc[ki][kj] = OperatorReduce::Apply(acc[ki][kj], OperatorMap::Apply(
//								abuf[ki], bbuf[kj]));
//					}
//				}
////				for (uint_t ai = k * size_m + i, bi = k * size_n + j, ki = 0; ki < pack_m; ki++) {
////#pragma HLS UNROLL factor=unroll_m
////					for (uint_t kj = 0; kj < pack_n; kj++) {
////#pragma HLS UNROLL factor=unroll_n
////						acc[ki][kj] = OperatorReduce::Apply(acc[ki][kj], OperatorMap::Apply(
////								(T)a[ai][ki], (T)b[bi][kj]));
////					}
////				}
//			}
//			for (uint_t ci = i * pack_m * size_n + j, ki = 0; ki < pack_m; ki++) {
//#pragma HLS UNROLL factor=unroll_m
//				for (uint_t cj = ci + ki * size_n, kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//					c[cj][kj] = acc[ki][kj];
//				}
//			}
//		}
//	}
//}
//
//template<uint_t dup_func, typename T, uint_t pack_n = 1, uint_t unroll_n = 1,
//		typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>,
//		typename RAM_b, typename RAM_c>
//void map_reduce(T a, RAM_b b, RAM_c c, T bias,
//		uint_t size_m, uint_t size_k, uint_t size_n) {
//	typedef hlslib::DataPack<T, pack_n> row_t;
//	for (uint_t i = 0; i < size_m; i++) {
//		for (uint_t j = 0; j < size_n; j++) {
//			T acc[pack_n];
//#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=unroll_n dim=1
//			for (uint_t kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//				acc[kj] = bias;
//			}
//			for (uint_t k = 0; k < size_k; k++) {
//				T bbuf[pack_n];
//#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=unroll_n dim=1
//				for (uint_t bi = (i * size_k + k) * size_n + j, kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//					bbuf[kj] = b[bi][kj];
//				}
//				for (uint_t kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//					acc[kj] = OperatorReduce::Apply(acc[kj], bbuf[kj]);
//				}
//			}
//			for (uint_t kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//				acc[kj] = OperatorMap::Apply(a, acc[kj]);
//			}
//			for (uint_t ci = i * size_n + j, kj = 0; kj < pack_n; kj++) {
//#pragma HLS UNROLL factor=unroll_n
//				c[ci][kj] = acc[kj];
//			}
//		}
//	}
//}

template<typename T, uint_t pack_m = 1, uint_t unroll_m = 1, uint_t pack_n = 1, uint_t unroll_n = 1,
		typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>,
		typename RAM_a, typename RAM_b, typename RAM_c0, typename RAM_c>
void matmul_acc_transpose_a(RAM_a a, RAM_b b, RAM_c0 c0, RAM_c c,
		uint_t size_m, uint_t size_k, uint_t size_n) {
	using col_t = hlslib::DataPack<T, pack_m>;
	using row_t = hlslib::DataPack<T, pack_n>;
	const uint_t pack_m_size_n = pack_m * size_n;
	for (uint_t off_c_row = 0, i = 0; i < size_m; i++, off_c_row += pack_m_size_n) {
#pragma HLS UNROLL factor=1
		for (uint_t j = 0; j < size_n; j++) {
#pragma HLS UNROLL factor=1
			row_t acc[pack_m];
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=unroll_m
			T acc_part[pack_m][pack_n];
#pragma HLS ARRAY_PARTITION variable=acc_part cyclic factor=unroll_m dim=1
#pragma HLS ARRAY_PARTITION variable=acc_part cyclic factor=unroll_n dim=2
			for (uint_t off_c = off_c_row + j, ki = 0; ki < pack_m; ki++, off_c += size_n) {
#pragma HLS UNROLL factor=1
				acc[ki] = c0[off_c];
			}
			for (uint_t ki = 0; ki < pack_m; ki++) {
#pragma HLS UNROLL factor=unroll_m
				acc[ki] >> acc_part[ki];
			}
			for (uint_t off_a = i, off_b = j,
					k = 0; k < size_k; k++,
					off_a += size_m, off_b += size_n) {
#pragma HLS UNROLL factor=1
				const col_t abuf = a[off_a];
				const row_t bbuf = b[off_b];
				T abuf_part[pack_m], bbuf_part[pack_n];
#pragma HLS ARRAY_PARTITION variable=abuf_part cyclic factor=unroll_m
#pragma HLS ARRAY_PARTITION variable=bbuf_part cyclic factor=unroll_n
				abuf >> abuf_part;
				bbuf >> bbuf_part;
				for (uint_t ki = 0; ki < pack_m; ki++) {
#pragma HLS UNROLL factor=unroll_m
					for (uint_t kj = 0; kj < pack_n; kj++) {
#pragma HLS UNROLL factor=unroll_n
						acc_part[ki][kj] = OperatorReduce::Apply(acc_part[ki][kj], OperatorMap::Apply(
								abuf_part[ki], bbuf_part[kj]));
					}
				}
			}
			for (uint_t ki = 0; ki < pack_m; ki++) {
#pragma HLS UNROLL factor=unroll_m
				acc[ki] << acc_part[ki];
			}
			for (uint_t off_c = off_c_row + j, ki = 0; ki < pack_m; ki++, off_c += size_n) {
#pragma HLS UNROLL factor=1
				c[off_c] = acc[ki];
			}
		}
	}
}

template<typename T, uint_t pack_m = 1, uint_t unroll_m = 1, uint_t pack_n = 1, uint_t unroll_n = 1,
		typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>,
		typename RAM_a, typename RAM_b, typename RAM_c>
void matmul_acc_transpose_a(RAM_a a, RAM_b b, RAM_c c,
		uint_t size_m, uint_t size_k, uint_t size_n) {
	matmul_acc_transpose_a<T, pack_m, unroll_m, pack_n, unroll_n, OperatorMap, OperatorReduce>(
			a, b, c, c, size_m, size_k, size_n);
}

template<typename T, uint_t pack_m, uint_t unroll_m, uint_t pack_n, uint_t unroll_n,
		typename RAM_c0>
void matmul_acc_transpose_a_dataflow_read(bool valid, RAM_c0 c0, T acc_part[pack_m][pack_n],
		uint_t size_n, uint_t off_c_start) {
#pragma HLS INLINE off
	using row_t = hlslib::DataPack<T, pack_n>;
	if (valid) {
		row_t acc[pack_m];
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=unroll_m
		for (uint_t off_c = off_c_start, ki = 0; ki < pack_m; ki++, off_c += size_n) {
#pragma HLS UNROLL factor=1
			acc[ki] = c0[off_c];
		}
		for (uint_t ki = 0; ki < pack_m; ki++) {
#pragma HLS UNROLL factor=unroll_m
			acc[ki] >> acc_part[ki];
		}
	}
}
template<typename T, uint_t pack_m, uint_t unroll_m, uint_t pack_n, uint_t unroll_n,
		typename OperatorMap, typename OperatorReduce,
		typename RAM_a, typename RAM_b>
void matmul_acc_transpose_a_dataflow_compute(bool valid, RAM_a a, RAM_b b, T acc_part[pack_m][pack_n],
		uint_t size_m, uint_t size_k, uint_t size_n,
		uint_t i, uint_t j) {
#pragma HLS INLINE off
	using col_t = hlslib::DataPack<T, pack_m>;
	using row_t = hlslib::DataPack<T, pack_n>;
	if (valid) {
		for (uint_t off_a = i, off_b = j,
				k = 0; k < size_k; k++,
				off_a += size_m, off_b += size_n) {
#pragma HLS UNROLL factor=1
			const col_t abuf = a[off_a];
			const row_t bbuf = b[off_b];
			T abuf_part[pack_m], bbuf_part[pack_n];
#pragma HLS ARRAY_PARTITION variable=abuf_part cyclic factor=unroll_m
#pragma HLS ARRAY_PARTITION variable=bbuf_part cyclic factor=unroll_n
			abuf >> abuf_part;
			bbuf >> bbuf_part;
			for (uint_t ki = 0; ki < pack_m; ki++) {
#pragma HLS UNROLL factor=unroll_m
				for (uint_t kj = 0; kj < pack_n; kj++) {
#pragma HLS UNROLL factor=unroll_n
					acc_part[ki][kj] = OperatorReduce::Apply(acc_part[ki][kj], OperatorMap::Apply(
							abuf_part[ki], bbuf_part[kj]));
				}
			}
		}
	}
}
template<typename T, uint_t pack_m, uint_t unroll_m, uint_t pack_n, uint_t unroll_n,
		typename RAM_c>
void matmul_acc_transpose_a_dataflow_write(bool valid, RAM_c c, T acc_part[pack_m][pack_n],
		uint_t size_n, uint_t off_c_start) {
#pragma HLS INLINE off
	using row_t = hlslib::DataPack<T, pack_n>;
	if (valid) {
		row_t acc[pack_m];
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=unroll_m
		for (uint_t ki = 0; ki < pack_m; ki++) {
#pragma HLS UNROLL factor=unroll_m
			acc[ki] << acc_part[ki];
		}
		for (uint_t off_c = off_c_start, ki = 0; ki < pack_m; ki++, off_c += size_n) {
#pragma HLS UNROLL factor=1
			c[off_c] = acc[ki];
		}
	}
}
template<typename T, uint_t pack_m, uint_t unroll_m, uint_t pack_n, uint_t unroll_n,
		typename RAM_c0, typename RAM_c>
void matmul_acc_transpose_a_dataflow_mem(bool valid_r, bool valid_w,
		RAM_c0 c0, RAM_c c, T acc_part_r[pack_m][pack_n], T acc_part_w[pack_m][pack_n],
		uint_t size_n, uint_t off_c_start_r, uint_t off_c_start_w) {
#pragma HLS INLINE off
	matmul_acc_transpose_a_dataflow_read<T, pack_m, unroll_m, pack_n, unroll_n>(valid_r,
			c0, acc_part_r, size_n, off_c_start_r);
	matmul_acc_transpose_a_dataflow_write<T, pack_m, unroll_m, pack_n, unroll_n>(valid_w,
			c, acc_part_w, size_n, off_c_start_w);
}

template<typename T, uint_t pack_m = 1, uint_t unroll_m = 1, uint_t pack_n = 1, uint_t unroll_n = 1,
		typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>,
		typename RAM_a, typename RAM_b, typename RAM_c0, typename RAM_c>
void matmul_acc_transpose_a_dataflow(RAM_a a, RAM_b b, RAM_c0 c0, RAM_c c,
		uint_t size_m, uint_t size_k, uint_t size_n) {
	using col_t = hlslib::DataPack<T, pack_m>;
	using row_t = hlslib::DataPack<T, pack_n>;
	const uint_t pack_m_size_n = pack_m * size_n;
	uint_t i1 = 0, j1 = 0, j2 = 0, off_c_row2 = 0, off_c_row1 = 0;
	bool valid[3] = {false, false, true};
#pragma HLS ARRAY_PARTITION variable=valid complete
	for (uint_t off_c_row = 0, i = 0; ; i++, off_c_row += pack_m_size_n) {
#pragma HLS UNROLL factor=1
		for (uint_t j = 0; j < size_n; j++) {
#pragma HLS UNROLL factor=1
			T acc_part[3][pack_m][pack_n];
#pragma HLS ARRAY_PARTITION variable=acc_part complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_part cyclic factor=unroll_m dim=2
#pragma HLS ARRAY_PARTITION variable=acc_part cyclic factor=unroll_n dim=3
			valid[2] = i < size_m;
			if (!(valid[0] | valid[1] | valid[2]))
				return;
			matmul_acc_transpose_a_dataflow_mem<T, pack_m, unroll_m, pack_n, unroll_n>(valid[2], valid[0],
					c0, c, acc_part[2], acc_part[0], size_n, off_c_row + j, off_c_row2 + j2);
			matmul_acc_transpose_a_dataflow_compute<T, pack_m, unroll_m, pack_n, unroll_n,
					OperatorMap, OperatorReduce>(valid[1],
					a, b, acc_part[1], size_m, size_k, size_n, i1, j1);
			// shift
			j2 = j1;
			j1 = j;
			off_c_row2 = off_c_row1;
			off_c_row1 = off_c_row;
			i1 = i;
			for (uint_t k = 0; k < 2; k++) {
#pragma HLS UNROLL
#pragma HLS DEPENDENCE variable=acc_part inter false
#pragma HLS DEPENDENCE variable=valid inter false
				for (uint_t ki = 0; ki < pack_m; ki++) {
#pragma HLS UNROLL
					for (uint_t kj = 0; kj < pack_n; kj++) {
#pragma HLS UNROLL
						acc_part[k][ki][kj] = acc_part[k + 1][ki][kj];
					}
				}
				valid[k] = valid[k + 1];
			}
		}
	}
}

template<typename T, uint_t pack_m = 1, uint_t unroll_m = 1, uint_t pack_n = 1, uint_t unroll_n = 1,
		typename OperatorMap = hlslib::op::Product<T>, typename OperatorReduce = hlslib::op::Sum<T>,
		typename RAM_a, typename RAM_b, typename RAM_c>
void matmul_acc_transpose_a_dataflow(RAM_a a, RAM_b b, RAM_c c,
		uint_t size_m, uint_t size_k, uint_t size_n) {
	matmul_acc_transpose_a_dataflow<T, pack_m, unroll_m, pack_n, unroll_n, OperatorMap, OperatorReduce>(
			a, b, c, c, size_m, size_k, size_n);
}

#endif

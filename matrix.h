#ifndef __CNN_HLS_MATRIX_H__
#define __CNN_HLS_MATRIX_H__
#include "types.h"
#include <hlslib/xilinx/DataPack.h>

template<typename T, uint_t pack_n = 1,
		typename RAM_x, typename RAM_y>
void copy_matrix(RAM_x src, RAM_y dst,
		uint_t src_num_cols, uint_t dst_num_cols,
		uint_t src_start, uint_t dst_start,
		uint_t block_rows, uint_t block_cols) {
//#pragma HLS INLINE
#pragma HLS INLINE OFF
	for (uint_t
			off_x_i = src_start,
			off_y_i = dst_start,
			i = 0; i < block_rows; i++,
			off_x_i += src_num_cols, off_y_i += dst_num_cols) {
		for (uint_t j = 0; j < block_cols; j++) {
			dst[off_y_i + j] = src[off_x_i + j];
		}
	}
}

template<typename T, uint_t pack_n = 1>
void pack_matrix(hlslib::DataPack<T, pack_n> *pack, const T (*mem)[pack_n],
		uint_t pack_num_cols, uint_t mem_num_cols,
		uint_t pack_start, uint_t mem_start,
		uint_t block_rows, uint_t block_cols) {
//#pragma HLS INLINE
#pragma HLS INLINE OFF
	for (uint_t
			off_x_i = mem_start,
			off_y_i = pack_start,
			i = 0; i < block_rows; i++,
			off_x_i += mem_num_cols, off_y_i += pack_num_cols) {
		for (uint_t j = 0; j < block_cols; j++) {
			pack[off_y_i + j] << mem[off_x_i + j];
		}
	}
}

template<typename T, uint_t pack_n = 1>
void unpack_matrix(const hlslib::DataPack<T, pack_n> *pack, T (*mem)[pack_n],
		uint_t pack_num_cols, uint_t mem_num_cols,
		uint_t pack_start, uint_t mem_start,
		uint_t block_rows, uint_t block_cols) {
//#pragma HLS INLINE
#pragma HLS INLINE OFF
	for (uint_t
			off_x_i = mem_start,
			off_y_i = pack_start,
			i = 0; i < block_rows; i++,
			off_x_i += mem_num_cols, off_y_i += pack_num_cols) {
		for (uint_t j = 0; j < block_cols; j++) {
			pack[off_y_i + j] >> mem[off_x_i + j];
		}
	}
}

template<typename T>
void copy_buffer(T *dst, const T *src, uint_t n) {
//#pragma HLS INLINE
#pragma HLS INLINE OFF
	for (uint_t i = 0; i < n; i++) {
		dst[i] = src[i];
	}
}

template<typename T, uint_t pack_n = 1,
		typename RAM_x>
void fill_matrix(RAM_x x, T value,
		uint_t num_cols,
		uint_t start,
		uint_t block_rows, uint_t block_cols) {
//#pragma HLS INLINE
#pragma HLS INLINE OFF
	using row_t = hlslib::DataPack<T, pack_n>;
	row_t t = value;
	for (uint_t off_x_i = start,
			i = 0; i < block_rows; i++,
			off_x_i += num_cols) {
		for (uint_t j = 0; j < block_cols; j++) {
			x[off_x_i + j] = t;
		}
	}
}

template<typename T, uint_t pack_n>
void load_row_bias_one_row(hlslib::DataPack<T, pack_n> *y, const T *bias,
		uint_t block_n,
		uint_t bias_i, uint_t y_off_i) {
//#pragma HLS INLINE
#pragma HLS INLINE OFF
	using row_t = hlslib::DataPack<T, pack_n>;
	row_t rowbuf = bias[bias_i];
	for (uint_t j = 0; j < block_n; j++) {
#pragma HLS UNROLL factor=1
		y[y_off_i + j] = rowbuf;
	}
}

template<typename T, uint_t pack_n>
void load_row_bias(hlslib::DataPack<T, pack_n> *y, const T *bias,
		uint_t num_cols,
		uint_t y_start, uint_t bias_start,
		uint_t block_m, uint_t block_n) {
//#pragma HLS INLINE
#pragma HLS INLINE OFF
	using row_t = hlslib::DataPack<T, pack_n>;
	for (uint_t y_off_i = y_start, i = 0; i < block_m; i++, y_off_i += num_cols) {
#pragma HLS UNROLL factor=1
		/*
		 * BUG..
		 * inlining this function will result in freezing
		 * hls reads bias in a burst of block_m but these values are not saved
		 */
		load_row_bias_one_row<T, pack_n>(
				y, bias, block_n,
				bias_start + i, y_off_i);
	}
}

template<typename T, uint_t pack_n>
void load_weight(hlslib::DataPack<T, pack_n> *a, const T *weight,
		uint_t a_num_cols, uint_t w_num_cols,
		uint_t a_start, uint_t w_start,
		uint_t block_m, uint_t block_n) {
#pragma HLS INLINE off
	for (uint_t
			off_x_i = w_start,
			off_y_i = a_start,
			i = 0; i < block_m; i++,
			off_x_i += w_num_cols, off_y_i += a_num_cols) {
		for (uint_t off_y = off_y_i, j = 0; j < block_n; j += pack_n, off_y++) {
			hlslib::DataPack<T, pack_n> row;
			for (uint_t k = 0; k < pack_n && j + k < block_n; k++) {
#pragma HLS UNROLL factor=1
				row[k] = weight[off_x_i + j + k];
			}
			a[off_y] = row;
		}
	}
}

#endif

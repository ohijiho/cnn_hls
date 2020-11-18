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
#pragma HLS INLINE
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

template<typename T, uint_t pack_n = 1,
		typename RAM_x, typename RAM_y>
void load_matrix(RAM_x mem, RAM_y buf,
		uint_t mem_cols,
		uint_t start,
		uint_t buf_rows, uint_t buf_cols) {
#pragma HLS INLINE
	for (uint_t
			off_x_i = start,
			off_y = 0,
			i = 0; i < buf_rows; i++,
			off_x_i += mem_cols) {
		for (uint_t j = 0; j < buf_cols; j++, off_y++) {
			buf[off_y] = mem[off_x_i + j];
		}
	}
}

template<typename T, uint_t pack_n = 1,
		typename RAM_x, typename RAM_y>
void store_matrix(RAM_x mem, RAM_y buf,
		uint_t mem_cols,
		uint_t start,
		uint_t buf_rows, uint_t buf_cols) {
#pragma HLS INLINE
	for (uint_t
			off_x_i = start,
			off_y = 0,
			i = 0; i < buf_rows; i++,
			off_x_i += mem_cols) {
		for (uint_t j = 0; j < buf_cols; j++, off_y++) {
			mem[off_x_i + j] = buf[off_y];
		}
	}
}

template<typename T, uint_t pack_n = 1,
		typename RAM_x>
void fill_matrix(RAM_x x, T value,
		uint_t num_cols,
		uint_t start,
		uint_t block_rows, uint_t block_cols) {
#pragma HLS INLINE
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

template<typename T, uint_t pack_m = 1, uint_t pack_n,
		typename RAM_y, typename RAM_bias>
void load_row_bias(RAM_y y, RAM_bias bias,
		uint_t num_cols,
		uint_t y_start, uint_t bias_start,
		uint_t block_m, uint_t block_n) {
#pragma HLS INLINE
	using col_t = hlslib::DataPack<T, pack_m>;
	using row_t = hlslib::DataPack<T, pack_n>;
	for (uint_t y_off_i = y_start, i = 0; i < block_m; i++) {
#pragma HLS UNROLL factor=1
		const col_t biasbuf = bias[bias_start + i];
		for (uint_t ki = 0; ki < pack_m; ki++, y_off_i += num_cols) {
#pragma HLS UNROLL factor=1
			row_t rowbuf = biasbuf[ki];
			for (uint_t j = 0; j < block_n; j++) {
#pragma HLS UNROLL factor=1
				y[y_off_i + j] = rowbuf;
			}
		}
	}
}

#endif

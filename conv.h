#ifndef __CNN_HLS_CONV_H__
#define __CNN_HLS_CONV_H__
#include "types.h"
#include "matrix.h"
#include "matmul.h"
#include <hls_math.h>
#include "im2col.h"
#include "utils.h"
#include <hlslib/xilinx/Operators.h>
#include "operators.h"

template<uint_t batch_size, typename T, uint_t pack_w,
		uint_t block_m, uint_t block_k, uint_t block_n>
class cnn_Conv2d_rtl_dataflow_class {
public:
	using row_t = hlslib::DataPack<T, batch_size>;
	using col_t = hlslib::DataPack<T, pack_w>;
	using im2col_t = iter_im2col<pack_w, batch_size, T>;
	static constexpr uint_t block_m_per_pack = block_m / pack_w;
private:
	im2col_t im2col;
	col_t a_bram[2][block_m_per_pack * block_k];
	row_t b_bram[3][block_k * block_n];
	row_t c_bram[3][block_m * block_n];
	bool nstop[3];
	decltype(im2col_t::last_result) resff[3];
	uint_t i, c_i, off_c_i;
	uint_t c_cur_size_m;
public:
	cnn_Conv2d_rtl_dataflow_class(size2_t input_size,
			uint_t in_channels, uint_t out_channels,
			size2_t kernel_size, size2_t stride, size2_t padding, size2_t dilation) :
				im2col(input_size, in_channels, out_channels, kernel_size, stride, padding, dilation,
						block_k, block_n, true),
				i(0), c_i(0), off_c_i(0), c_cur_size_m(0) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=a_bram dim=1 complete
#pragma HLS ARRAY_PARTITION variable=b_bram dim=1 complete
#pragma HLS ARRAY_PARTITION variable=c_bram dim=1 complete
#pragma HLS ARRAY_PARTITION variable=nstop dim=1 complete
#pragma HLS ARRAY_PARTITION variable=resff dim=1 complete
	}
private:
	template<typename RAM_x>
	void stage0(RAM_x x) {
#pragma HLS INLINE OFF
		if (nstop[2]) {
			im2col.dump(x, b_bram[2]);
		}
	}
	template<typename RAM_y, typename RAM_weight, typename RAM_bias>
	void stage1(RAM_y y, RAM_weight weight, RAM_bias bias) {
#pragma HLS INLINE OFF
		const auto &res = resff[2];
		if (nstop[2]) {
			if (res.valid & res.weight) {
				load_weight<T, pack_w>(a_bram[1], weight,
						block_m_per_pack, im2col.out_channels,
						0, res.k * im2col.out_channels + c_i,
						res.size_k, c_cur_size_m);
			}
			if (res.valid & (res.bias | res.c_read)) {
				if (res.bias) {
					load_row_bias<T, batch_size>(c_bram[2], bias,
							block_n,
							0, c_i,
							c_cur_size_m, res.size_n);
				} else {
					pack_matrix<T, batch_size>(c_bram[2], y,
							block_n, im2col.size_n,
							0, off_c_i + res.j,
							c_cur_size_m, res.size_n);
				}
			}
		}
	}
	void stage2() {
#pragma HLS INLINE OFF
		const auto &res = resff[1];
		if (nstop[1]) {
			matmul_acc_transpose_a_dataflow<T, pack_w, pack_w, batch_size, batch_size>(
					a_bram[0], b_bram[0], c_bram[1], c_bram[1], block_m_per_pack, res.size_k, block_n);
		}
	}
	template<typename RAM_y>
	void stage3(RAM_y y) {
#pragma HLS INLINE OFF
		const auto &res = resff[0];
		if (res.valid & res.c_write) {
			unpack_matrix<T, batch_size>(c_bram[0], y,
					block_n, im2col.size_n,
					0, off_c_i + res.j,
					c_cur_size_m, res.size_n);
		}
	}
	template<typename RAM_x, typename RAM_y, typename RAM_weight, typename RAM_bias>
	void loop_body(RAM_x x, RAM_y y, RAM_weight weight, RAM_bias bias) {
//#pragma HLS DATAFLOW
		stage0(x);
		stage1(y, weight, bias);
		stage2();
		stage3(y);
	}
public:
	template<typename RAM_x, typename RAM_y, typename RAM_weight, typename RAM_bias>
	void do_loop(RAM_x x, RAM_y y, RAM_weight weight, RAM_bias bias) {
		for (off_c_i = 0, c_i = 0,
				i = 0; i < im2col.size_m; i += block_m_per_pack,
				off_c_i += block_m * im2col.size_n, c_i += block_m) {
			c_cur_size_m = std::min(block_m, im2col.out_channels - c_i);
			nstop[0] = true;
			nstop[1] = true;
			nstop[2] = true;
			resff[0] = {false,};
			resff[1] = {false,};
			resff[2] = {false,};
			for (im2col.reset(); nstop[0];) {
				loop_body(x, y, weight, bias);
				copy_buffer(a_bram[0], a_bram[1], block_m_per_pack * block_k);
				for (uint_t k = 0; k < 2; k++) {
					copy_buffer(b_bram[k], b_bram[k + 1], block_k * block_n);
				}
				for (uint_t k_end = resff[2].valid & (resff[2].bias | resff[2].c_read) ? 2 : 1,
						k = 0; k < k_end; k++) {
					copy_buffer(c_bram[k], c_bram[k + 1], block_m * block_n);
				}
//				for (uint_t k = 0; k < resff[2].valid & (resff[2].bias | resff[2].c_read) ? 2 : 1; k++) {
//					copy_buffer(c_bram[k], c_bram[k + 1], block_m * block_n);
//				}
				resff[0] = resff[1],
				resff[1] = resff[2],
				resff[2] = im2col.last_result,
				nstop[0] = nstop[1],
				nstop[1] = nstop[2],
				nstop[2] = im2col.last_result.valid;
			}
		}
	}
};

template<uint_t batch_size, typename T, uint_t pack_w = 1,
		typename RAM_x, typename RAM_y, typename RAM_weight, typename RAM_bias>
void cnn_Conv2d_rtl_dataflow(RAM_x x, RAM_y y, RAM_weight weight, RAM_bias bias,
		size2_t input_size,
		uint_t in_channels, uint_t out_channels,
		size2_t kernel_size, size2_t stride, size2_t padding, size2_t dilation) {
//	const uint_t block_m = (5 - 1) / pack_w + 1, block_k = 5, block_n = 4;
	constexpr uint_t block_m = ((5 - 1) / pack_w + 1) * pack_w, block_k = 8, block_n = 32; // result in brams of 256 elements
	constexpr uint_t block_m_per_pack = block_m / pack_w;
	static_assert(block_m_per_pack * pack_w == block_m, "block_m must be a multiple of pack_w");
	cnn_Conv2d_rtl_dataflow_class<batch_size, T, pack_w, block_m, block_k, block_n> conv(
			input_size, in_channels, out_channels, kernel_size, stride, padding, dilation);
	conv.do_loop(x, y, weight, bias);
}

template<uint_t batch_size, typename T, uint_t pack_w,
		uint_t block_m, uint_t block_k, uint_t block_n>
class cnn_Conv2d_hls_dataflow_class {
public:
	using row_t = hlslib::DataPack<T, batch_size>;
	using col_t = hlslib::DataPack<T, pack_w>;
	using im2col_t = iter_im2col<pack_w, batch_size, T>;
	static constexpr uint_t block_m_per_pack = block_m / pack_w;
private:
	im2col_t im2col;
	col_t a_bram[block_m_per_pack * block_k];
	row_t b_bram[block_k * block_n];
	row_t c0_bram[block_m * block_n];
	row_t c_bram[block_m * block_n];
	row_t c_fb_bram[block_m * block_n];
	uint_t i, off_c_i;
	bool nstop;
	uint_t cur_size_m;
public:
	cnn_Conv2d_hls_dataflow_class(size2_t input_size,
			uint_t in_channels, uint_t out_channels,
			size2_t kernel_size, size2_t stride, size2_t padding, size2_t dilation) :
				im2col(input_size, in_channels, out_channels, kernel_size, stride, padding, dilation,
						block_k, block_n, true),
				i(0), off_c_i(0), cur_size_m(0), nstop(false) {
#pragma HLS INLINE
	}
private:
	template<typename RAM_x>
	void stage0(RAM_x x) {
		if (nstop) {
			nstop = im2col.dump(x, b_bram);
		}
	}
	template<typename RAM_y, typename RAM_weight, typename RAM_bias>
	void stage1(RAM_y y, RAM_weight weight, RAM_bias bias) {
		const auto &res = im2col.last_result;
		if (res.valid) {
			if (res.weight) {
				copy_matrix<T, pack_w>(weight, a_bram,
						im2col.size_m, block_m,
						res.k * im2col.size_m + i, 0,
						res.size_k, cur_size_m);
				fill_matrix<T, pack_w>(a_bram, (T)0, block_m, res.size_k * block_m, block_k - res.size_k, cur_size_m);
			}
			if (res.bias | res.c_read) {
				if (res.bias) {
					load_row_bias<T, pack_w, batch_size>(c0_bram, bias,
							block_n,
							0, i,
							cur_size_m, res.size_n);
				}
				else {
					copy_matrix<T, batch_size>(y, c0_bram,
							im2col.size_n, block_n,
							off_c_i + res.j, 0,
							cur_size_m, res.size_n);
				}
			} else {
				copy_buffer(c0_bram, c_fb_bram, block_m * block_n);
			}
		}
	}
	void stage2() {
		if (im2col.last_result) {
			matmul_acc_transpose_a<T, pack_w, pack_w, batch_size, batch_size>(
					a_bram, b_bram, c0_bram, c_bram, block_m_per_pack, block_k, block_n);
		}
	}
	template<typename RAM_y>
	void stage3(RAM_y y) {
		const auto &res = im2col.last_result;
		if (res.valid) {
			if (res.c_write) {
				copy_matrix<T, batch_size>(c_bram, y,
						block_n, im2col.size_n,
						0, off_c_i + res.j,
						cur_size_m, res.size_n);
			} else {
				copy_buffer(c_fb_bram, c_bram, block_m * block_n);
			}
		}
	}
public:
	template<typename RAM_x, typename RAM_y, typename RAM_weight, typename RAM_bias>
	void do_loop(RAM_x x, RAM_y y, RAM_weight weight, RAM_bias bias) {
		for (off_c_i = 0, i = 0; i < im2col.size_m; i += block_m, off_c_i += block_m * im2col.size_n) {
			cur_size_m = std::min(block_m, im2col.size_m - i);
			nstop = true;
			im2col.reset();
			const uint_t bound = bound = im2col.max_iter;
			for (uint_t counter = 0; counter < bound; counter++) {
#pragma HLS DATAFLOW
				/*
				 * IMPOSSIBLE:
				 * 	dataflow with feedback in HLS is impossible
				 * 	going back to RTL approach
				 */
				stage0(x);
				stage1(y, weight, bias);
				stage2();
				stage3(y);
			}
		}
	}
};

template<uint_t batch_size, typename T, uint_t pack_w = 1,
		typename RAM_x, typename RAM_y, typename RAM_weight, typename RAM_bias>
void cnn_Conv2d_hls_dataflow(RAM_x x, RAM_y y, RAM_weight weight, RAM_bias bias,
		size2_t input_size,
		uint_t in_channels, uint_t out_channels,
		size2_t kernel_size, size2_t stride, size2_t padding, size2_t dilation) {
//	const uint_t block_m = (5 - 1) / pack_w + 1, block_k = 5, block_n = 4;
	constexpr uint_t block_m = 8, block_k = 8, block_n = 32; // result in brams of 256 elements
	constexpr uint_t block_m_per_pack = block_m / pack_w;
	static_assert(block_m_per_pack * pack_w == block_m, "block_m must be a multiple of pack_w");
	cnn_Conv2d_hls_dataflow_class<batch_size, T, pack_w, block_m, block_k, block_n> conv(
			input_size, in_channels, out_channels, kernel_size, stride, padding, dilation);
	conv.do_loop(x, y, weight, bias);
}

#endif

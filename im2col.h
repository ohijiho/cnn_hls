#ifndef __CNN_HLS_IM2COL_H__
#define __CNN_HLS_IM2COL_H__

#include "types.h"
#include "utils.h"
#include <sstream>
#include <hlslib/xilinx/DataPack.h>

template<uint_t batch_size, typename T,
		typename RAM_x>
class ram_im2col {
public:
	typedef hlslib::DataPack<T, batch_size> row_t;

private:
	ram_im2col(RAM_x x,
			size2_t input_size,
			uint_t in_channels, uint_t out_channels,
			size2_t kernel_size, size2_t stride, size2_t padding, size2_t dilation,
			T pad_value, size2_t output_size)
			: ram(x),
			  input_size(input_size),
			  in_channels(in_channels), out_channels(out_channels),
			  kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation),
			  output_size(output_size),
			  pad_value(pad_value),
			  size_m(out_channels), size_k(in_channels * kernel_size.area()), size_n(output_size.area()) {
#pragma HLS INLINE
	}
public:
	ram_im2col(RAM_x x,
			size2_t input_size,
			uint_t in_channels, uint_t out_channels,
			size2_t kernel_size, size2_t stride, size2_t padding, size2_t dilation,
			T pad_value = 0)
			: ram_im2col(x,
					input_size,
					in_channels, out_channels,
					kernel_size, stride, padding, dilation,
					pad_value, calc_output_size(input_size, kernel_size, stride, padding, dilation)) {
#pragma HLS INLINE
	}

	row_t get(int_t ci, int_t ki, int_t oi) const {
#pragma HLS INLINE

		const int_t oci = oi % output_size.width;
		const int_t ori = oi / output_size.width;
		const int_t kci = ki % kernel_size.width;
		const int_t kri = ki / kernel_size.width;

		const int_t ici = oci * stride.width + kci * dilation.width - padding.width;
		const int_t iri = ori * stride.height + kri * dilation.height - padding.height;
		if (ici < 0 || ici >= (int_t)input_size.width || iri < 0 || iri >= (int_t)input_size.height)
			return pad_value;
		return ram[ci * input_size.area() + iri * input_size.width + ici];
	}
	row_t get(int_t i) const {
#pragma HLS INLINE
		const int_t oi = i % output_size.area();
		const int_t ki = i / output_size.area() % kernel_size.area();
		const int_t ci = i / output_size.area() / kernel_size.area();
		return this->get(ci, ki, oi);
	}

	row_t operator[](int_t i) const {
#pragma HLS INLINE
		return this->get(i);
	}

	const uint_t size_m, size_k, size_n;

	template <typename RAM_y>
	void dump_all(RAM_y y) const {
#pragma HLS INLINE
#if 0
		for (int_t i = 0; i < size_k * size_n; i++)
			y[i] = get(i);
#else
		const RAM_x x = ram;
		const uint_t dilation_hoff = dilation.height * input_size.width;
		const uint_t stride_hoff = stride.height * input_size.width;
		const uint_t padding_hoff = padding.height * input_size.width;
		const uint_t input_size_area = input_size.area();
		for (int_t ci = 0, off_y = 0, off_x_ch = 0; ci < (int_t)in_channels; ci++, off_x_ch += input_size_area) {
			for (int_t kri = 0, off_kri = 0; kri < (int_t)kernel_size.height; kri++, off_kri += dilation_hoff) {
				for (int_t kci = 0, off_kci = 0; kci < (int_t)kernel_size.width; kci++, off_kci += dilation.width) {
					for (int_t ori = 0, off_x_row = off_kri - padding_hoff; ori < (int_t)output_size.height; ori++, off_x_row += stride_hoff) {
						for (int_t oci = 0, off_x_col = off_kci - padding.width; oci < (int_t)output_size.width; oci++, off_x_col += stride.width, off_y++) {
							row_t t;
							if (
									off_x_row >= 0 && off_x_row < (int_t)input_size_area &&
									off_x_col >= 0 && off_x_col < (int_t)input_size.width) {
								t = x[off_x_ch + off_x_row + off_x_col];
							} else {
								t = pad_value;
							}
							y[off_y] = t;
						}
					}
				}
			}
		}
#endif
	}

private:
	const RAM_x ram;
	const size2_t input_size;
	const uint_t in_channels, out_channels;
	const size2_t kernel_size, stride, padding, dilation;
	const size2_t output_size;
	const row_t pad_value;
};

template<uint_t pack_w, uint_t batch_size, typename T>
class iter_im2col {
public:
	static const uint_t pack_m = pack_w;
	static const uint_t pack_n = batch_size;
	using col_t = hlslib::DataPack<T, pack_m>;
	using row_t = hlslib::DataPack<T, pack_n>;

	const size2_t input_size;
	const uint_t in_channels, out_channels;
	const size2_t kernel_size, stride, padding, dilation;
	const size2_t output_size;
	const uint_t block_k, block_n;
	const bool column_first;
	const row_t pad_value;

private:
	const uint_t dilation_hoff;
	const uint_t stride_hoff;
	const uint_t padding_hoff;
	const uint_t input_size_area;

public:
	const uint_t size_m, size_k, size_n;
	const uint_t max_iter;

private:
	const uint_t image_size;

private:
	struct {
		struct {
			int_t yi, off_x_ch;
			int_t kri, off_kri;
			int_t kci, off_kci;
		} i;
		struct {
			int_t yj, off_ori;
			int_t oci, off_oci;
		} j;
		int_t off_img;
	} state2;
	decltype(state2) const state2_init;

public:
	iter_im2col(
			size2_t input_size,
			uint_t in_channels, uint_t out_channels,
			size2_t kernel_size, size2_t stride, size2_t padding, size2_t dilation,
			uint_t block_k, uint_t block_n,
			bool column_first,
			T pad_value = 0)
			: input_size(input_size),
			  in_channels(in_channels), out_channels(out_channels),
			  kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation),
			  output_size(calc_output_size(input_size, kernel_size, stride, padding, dilation)),
			  block_k(block_k), block_n(block_n),
			  column_first(column_first),
			  pad_value(pad_value),
			  dilation_hoff(dilation.height * input_size.width),
			  stride_hoff(stride.height * input_size.width),
			  padding_hoff(padding.height * input_size.width),
			  input_size_area(input_size.area()),
			  size_m((out_channels - 1) / pack_m + 1),
			  size_k(in_channels * kernel_size.area()),
			  size_n(output_size.area()),
			  max_iter(((size_k - 1) / block_k + 1) * ((size_n - 1) / block_n + 1)),
			  image_size(size_k * size_n),
			  state2_init({
		.i = {
				.yi = 0, .off_x_ch = 0,
				.kri = 0, .off_kri = 0,
				.kci = 0, .off_kci = 0,
		},
		.j = {
				.yj = 0, .off_ori = -(int_t)padding_hoff,
				.oci = 0, .off_oci = -(int_t)padding.width,
		},
		.off_img = 0,
	}) {
		/*
		 * the order of initialization of fields is the same as definition.
		 * it is safe to initialize size_n with output_size
		 */
//#pragma HLS INLINE
#pragma HLS ALLOCATION instances=umul limit=1 operation
#pragma HLS ALLOCATION instances=udiv limit=1 operation
		state2 = state2_init;
		last_result.valid = false;
	}

	struct dump_result {
		bool valid, weight, bias, c_read, c_write;
		uint_t k, j, size_k, size_n;
		inline operator bool() {
#pragma HLS INLINE
			return valid;
		}
	} last_result;

	dump_result tell() {
//#pragma HLS INLINE
		auto &s = state2;
		if (column_first ? (s.j.yj == size_n) : (s.i.yi == size_k)) // iteration stop condition
			return {false, };
		const int_t yi_end = std::min(s.i.yi + block_k, size_k);
		const int_t yj_end = std::min(s.j.yj + block_n, size_n);
		return dump_result{
				.valid = true,
				.weight = column_first || s.j.yj == 0,
				.bias = s.i.yi == 0,
				.c_read = !column_first && s.i.yi != 0,
				.c_write = !column_first || yi_end == size_k,
				.k = (uint_t)s.i.yi,
				.j = (uint_t)s.j.yj,
				.size_k = (uint_t)(yi_end - s.i.yi),
				.size_n = (uint_t)(yj_end - s.j.yj),
		};
	}

	template <typename RAM_x, typename RAM_y>
	bool dump(RAM_x x, RAM_y y) {
//#pragma HLS INLINE
		{
			auto &s = state2;
			if (column_first ? (s.j.yj == size_n) : (s.i.yi == size_k)) // iteration stop condition
				return last_result = {false, };
			auto i = s.i;
			auto j = s.j;
			const int_t yi_end = std::min(s.i.yi + block_k, size_k);
			const int_t yj_end = std::min(s.j.yj + block_n, size_n);

			last_result.valid = true;
			last_result.weight = column_first || j.yj == 0;
			last_result.bias = i.yi == 0;
			last_result.c_read = !column_first && i.yi != 0;
			last_result.c_write = !column_first || yi_end == size_k;
			last_result.k = (uint_t)i.yi;
			last_result.j = (uint_t)j.yj;
			last_result.size_k = (uint_t)(yi_end - i.yi);
			last_result.size_n = (uint_t)(yj_end - j.yj);

			for (int_t off_y_row = 0;; i.yi++, i.kci++, i.off_kci += dilation.width, off_y_row += block_n) { // loop i3 increment
				if (i.kci == kernel_size.width) { // loop i3 stop condition
					i.kri++, i.off_kri += dilation_hoff; // loop i2 increment
					if (i.kri == kernel_size.height) { // loop i2 stop condition
						i.off_x_ch += input_size_area; // loop i1 increment
						i.kri = 0, i.off_kri = 0; // loop i2 initialization
					}
					i.kci = 0, i.off_kci = 0; // loop i3 initialization
				}
				if (i.yi == yi_end) // block bound
					break;
				j = s.j;
				for (int_t off_y = off_y_row;; j.yj++, j.oci++, j.off_oci += stride.width,
						off_y++) { // loop j2 increment
					if (j.oci == output_size.width) { // loop j2 stop condition
						j.off_ori += stride_hoff; // loop j1 increment
						j.oci = 0, j.off_oci = -padding.width; // loop j2 initialization
					}
					if (j.yj == yj_end) // block bound
						break;
					const int_t off_x_row = i.off_kri + j.off_ori;
					const int_t off_x_col = i.off_kci + j.off_oci;
					row_t t;
					if (
							off_x_row >= 0 && off_x_row < (int_t)input_size_area &&
							off_x_col >= 0 && off_x_col < (int_t)input_size.width) {
						t << x[s.off_img + i.off_x_ch + off_x_row + off_x_col];
					} else {
						t = pad_value;
					}
					y[off_y] = t;
				}
			}
			/*
			 * i, j point to next row and column
			 */
			if (column_first) {
				if (yi_end == size_k) { // end of column
					s.j = j;
					s.i = state2_init.i;
				} else {
					s.i = i; // choose next row
				}
			} else {
				if (yj_end == size_n) {
					s.i = i;
					s.j = state2_init.j;
				} else {
					s.j = j;
				}
			}
			return true;
		}
	}

	void reset() {
//#pragma HLS INLINE
		state2 = state2_init;
		last_result.valid = false;
	}
	void next_image() {
//#pragma HLS INLINE
		state2.i = state2_init.i;
		state2.j = state2_init.j;
		state2.off_img += image_size;
	}
};

#endif

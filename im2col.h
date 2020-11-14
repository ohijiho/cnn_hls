#ifndef __CNN_HLS_IM2COL_H__
#define __CNN_HLS_IM2COL_H__

#include "types.h"
#include "utils.h"
#include <sstream>
#include <hlslib/xilinx/DataPack.h>

template<size_t batch_size, size_t pack_out_channels, typename T,
		typename RAM_x>
class ram_im2col {
public:
	static constexpr size_t pack_n = batch_size;
	static constexpr size_t pack_m = pack_out_channels;
	typedef hlslib::DataPack<T, pack_n> row_t;
	typedef hlslib::DataPack<T, pack_m> col_t;
	typedef hlslib::DataPack<row_t, pack_m> mat_t;

private:
	ram_im2col(RAM_x x,
			size_2_t input_size,
			size_t in_channels, size_t out_channels,
			size_2_t kernel_size, size_2_t stride, size_2_t padding, size_2_t dilation,
			T pad_value, size_2_t output_size)
			: ram(x),
			  input_size(input_size),
			  in_channels(in_channels), out_channels(out_channels),
			  kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation),
			  output_size(output_size),
			  pad_value(pad_value),
			  size_m(out_channels), size_k(in_channels * kernel_size.area()), size_n(output_size.area()) {
#pragma HLS inline
	}
public:
	ram_im2col(RAM_x x,
			size_2_t input_size,
			size_t in_channels, size_t out_channels,
			size_2_t kernel_size, size_2_t stride, size_2_t padding, size_2_t dilation,
			T pad_value = 0)
			: ram_im2col(x,
					input_size,
					in_channels, out_channels,
					kernel_size, stride, padding, dilation,
					pad_value, calc_output_size(input_size, kernel_size, stride, padding, dilation)) {
#pragma HLS inline
	}

	row_t get(ptrdiff_t ci, ptrdiff_t ki, ptrdiff_t oi) const {
#pragma HLS inline

		const ptrdiff_t oci = oi % output_size.width;
		const ptrdiff_t ori = oi / output_size.width;
		const ptrdiff_t kci = ki % kernel_size.width;
		const ptrdiff_t kri = ki / kernel_size.width;

		const ptrdiff_t ici = oci * stride.width + kci * dilation.width - padding.width;
		const ptrdiff_t iri = ori * stride.height + kri * dilation.height - padding.height;
		if (ici < 0 || ici >= input_size.width || iri < 0 || iri >= input_size.height)
			return pad_value;
		return ram[ci * input_size.area() + iri * input_size.width + ici];
	}
	row_t get(ptrdiff_t i) const {
#pragma HLS inline
		const ptrdiff_t oi = i % output_size.area();
		const ptrdiff_t ki = i / output_size.area() % kernel_size.area();
		const ptrdiff_t ci = i / output_size.area() / kernel_size.area();
		return this->get(ci, ki, oi);
	}

	row_t operator[](ptrdiff_t i) const {
#pragma HLS inline
		return this->get(i);
	}

	const size_t size_m, size_k, size_n;

private:
	const RAM_x ram;
	const size_2_t input_size;
	const size_t in_channels, out_channels;
	const size_2_t kernel_size, stride, padding, dilation;
	const size_2_t output_size;
	const T pad_value;
};

template <size_t batch_size, typename T,
	typename RAM_x, typename RAM_y, typename RAM_weight, typename RAM_bias>
void dump_im2col(RAM_x x, RAM_y y,
		size_2_t input_size,
		size_t in_channels, size_t out_channels,
		size_2_t kernel_size, size_2_t stride, size_2_t padding, size_2_t dilation) {
#pragma HLS inline
	ram_im2col<batch_size, 1, T, RAM_x> im2col(x, input_size, in_channels, out_channels, kernel_size, stride, padding, dilation);
	for (ptrdiff_t i = 0; i < im2col.size_k * im2col.size_n; i++)
		y[i] = im2col[i];
}

#endif

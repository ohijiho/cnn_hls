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
	static constexpr uint_t pack_n = batch_size;
	typedef hlslib::DataPack<T, pack_n> row_t;

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
#pragma HLS inline
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
#pragma HLS inline
	}

	row_t get(int_t ci, int_t ki, int_t oi) const {
#pragma HLS inline

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
#pragma HLS inline
		const int_t oi = i % output_size.area();
		const int_t ki = i / output_size.area() % kernel_size.area();
		const int_t ci = i / output_size.area() / kernel_size.area();
		return this->get(ci, ki, oi);
	}

	row_t operator[](int_t i) const {
#pragma HLS inline
		return this->get(i);
	}

	const uint_t size_m, size_k, size_n;

private:
	const RAM_x ram;
	const size2_t input_size;
	const uint_t in_channels, out_channels;
	const size2_t kernel_size, stride, padding, dilation;
	const size2_t output_size;
	const T pad_value;
};

template <uint_t batch_size, typename T,
	typename RAM_x, typename RAM_y, typename RAM_weight, typename RAM_bias>
void dump_im2col(RAM_x x, RAM_y y,
		size2_t input_size,
		uint_t in_channels, uint_t out_channels,
		size2_t kernel_size, size2_t stride, size2_t padding, size2_t dilation) {
#pragma HLS inline
	ram_im2col<batch_size, T, RAM_x> im2col(x, input_size, in_channels, out_channels, kernel_size, stride, padding, dilation);
	for (int_t i = 0; i < im2col.size_k * im2col.size_n; i++)
		y[i] = im2col[i];
}

#endif

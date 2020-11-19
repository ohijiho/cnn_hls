#ifndef __CNN_HLS_UTILS_H__
#define __CNN_HLS_UTILS_H__
#include "types.h"
template <typename T>
constexpr T log2_floor(T x) {
	return x < 2 ? 0 : 1 + (log2_floor(x) >> 1);
}
template <typename T>
constexpr T log2_ceil(T x) {
	return x < 2 ? 0 : log2_floor(x - 1) + 1;
}
template <typename T>
constexpr T calc_output_size(T input_size, T kernel_size, T stride, T padding, T dilation) {
	return (input_size + 2 * padding - ((kernel_size - 1) * dilation) - 1) / stride + 1;
}

template <typename T, uint_t n, uint_t m>
void partial_unpack(hlslib::DataPack<T, m> dst[n / m], hlslib::DataPack<T, n> src) {
#pragma HLS INLINE
	T tmp[n];
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=m
	src >> tmp;
	for (uint_t i = 0; i < n / m; i++) {
		dst[i] << tmp + i * m;
	}
}
template <typename T, uint_t n, uint_t m>
hlslib::DataPack<T, n> partial_pack(const hlslib::DataPack<T, m> src[n / m]) {
#pragma HLS INLINE
	hlslib::DataPack<T, n> ret;
	T tmp[n];
#pragma HLS ARRAY_PARTITION variable=tmp cyclic factor=m
	for (uint_t i = 0; i < n / m; i++) {
		src[i] >> tmp + i * m;
	}
	ret << tmp;
	return ret;
}

#define DO_PRAGMA(x) _Pragma (#x)

#endif

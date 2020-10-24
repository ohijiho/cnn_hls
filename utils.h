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
#endif

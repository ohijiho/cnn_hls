#ifndef __CNN_HLS_TYPES_H__
#define __CNN_HLS_TYPES_H__
#include <stddef.h>
#include <stdint.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_math.h>

#include <sstream>
#define HLSLIB_SYNTHESIS
#include <hlslib/xilinx/DataPack.h>

#include "config.h"

typedef
#if VALUE_FLOAT
#if VALUE_SIZE == 32
		float
#elif VALUE_SIZE == 64
		double
#endif
#else
		ap_fixed<VALUE_SIZE, VALUE_INT_PART>
#endif
		value_t;

#define MACRO_CAT_3(a, b, c) a##b##c
#define MACRO_SUB_CAT_3(a, b, c) MACRO_CAT_3(a, b, c)
typedef
#if INT_PRIMITIVE
		MACRO_SUB_CAT_3(int, INT_SIZE, _t)
#else
		ap_int<INT_SIZE>
#endif
		int_t;
typedef
#if INT_PRIMITIVE
		MACRO_SUB_CAT_3(uint, INT_SIZE, _t)
#else
		ap_uint<INT_SIZE>
#endif
		uint_t;
#undef MACRO_CAT_3
#undef MACRO_SUB_CAT_3

#define PACK_W_SIZE(n) (((n) - 1) / PACK_W + 1)
#define ALIGN_W(n) (((n) - 1) / PACK_W + 1)

typedef hlslib::DataPack<value_t, BATCH_SIZE> minibatch_t;
typedef hlslib::DataPack<value_t, PACK_W> weight_t;

struct size2_t {
	uint_t width, height;
	constexpr uint_t area() const {
#pragma HLS INLINE
		return height * width;
	}
};

#define TYPES_SIZE_2_T_BINARY_OP(op, inplace) \
inline size2_t operator op(size2_t a, size2_t b) {\
_Pragma("HLS INLINE")\
	return {a.width op b.width, a.height op b.height};\
}\
inline size2_t operator op(size2_t a, uint_t b) {\
	_Pragma("HLS INLINE")\
	return {a.width op b, a.height op b};\
}\
inline size2_t operator op(uint_t a, size2_t b) {\
	_Pragma("HLS INLINE")\
	return {a op b.width, a op b.height};\
}\
inline size2_t &operator inplace(size2_t &a, size2_t b) {\
	_Pragma("HLS INLINE")\
	a.width inplace b.width;\
	a.height inplace b.height;\
	return a;\
}\
inline size2_t &operator inplace(size2_t &a, uint_t b) {\
	_Pragma("HLS INLINE")\
	a.width inplace b;\
	a.height inplace b;\
	return a;\
}
TYPES_SIZE_2_T_BINARY_OP(+, +=)
TYPES_SIZE_2_T_BINARY_OP(-, -=)
TYPES_SIZE_2_T_BINARY_OP(*, *=)
TYPES_SIZE_2_T_BINARY_OP(/, /=)
TYPES_SIZE_2_T_BINARY_OP(%, %=)
TYPES_SIZE_2_T_BINARY_OP(^, ^=);
TYPES_SIZE_2_T_BINARY_OP(|, |=);
TYPES_SIZE_2_T_BINARY_OP(&, &=);
#undef TYPES_SIZE_2_T_BINARY_OP

#define UNPACK_SIZE2(sz) (sz).width, (sz).height

template<typename T>
class MaxIdentity {
public:
	static constexpr T value() { return std::numeric_limits<T>::lowest(); }
};
template<>
class MaxIdentity<float> {
public:
	static constexpr float value() { return -INFINITY; }
};
template<>
class MaxIdentity<double> {
public:
	static constexpr double value() { return -INFINITY; }
};
template<int _AP_W, int _AP_I, ap_q_mode _AP_Q, ap_o_mode _AP_O, int _AP_N>
class MaxIdentity<ap_fixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>> {
	using T = ap_fixed<_AP_W, _AP_I, _AP_Q, _AP_O, _AP_N>;
public:
	static constexpr T value() { return T(1) << (_AP_I - 1); }
};
template<int _AP_W>
class MaxIdentity<ap_int<_AP_W>> {
	using T = ap_int<_AP_W>;
public:
	static constexpr T value() { return T(1) << (_AP_W - 1); }
};

#endif

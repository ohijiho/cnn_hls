#ifndef __CNN_HLS_TYPES_H__
#define __CNN_HLS_TYPES_H__
#include <stddef.h>
#include <stdint.h>
#include <ap_int.h>
#include <ap_fixed.h>

#include <sstream>
#define HLSLIB_SYNTHESIS
#include <hlslib/xilinx/DataPack.h>

#include "config.h"

struct size_2_t {
	size_t width, height;
	size_t area() const { return height * width; }
};

#define TYPES_SIZE_2_T_BINARY_OP(op, inplace) \
inline size_2_t operator op(size_2_t a, size_2_t b) {\
	return {a.width op b.width, a.height op b.height};\
}\
inline size_2_t operator op(size_2_t a, size_t b) {\
	return {a.width op b, a.height op b};\
}\
inline size_2_t operator op(size_t a, size_2_t b) {\
	return {a op b.width, a op b.height};\
}\
inline size_2_t &operator inplace(size_2_t &a, size_2_t b) {\
	a.width inplace b.width;\
	a.height inplace b.height;\
	return a;\
}\
inline size_2_t &operator inplace(size_2_t &a, size_t b) {\
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

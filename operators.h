#ifndef __CNN_HLS_OPERATORS_H__
#define __CNN_HLS_OPERATORS_H__

template <typename key_t, typename value_t>
struct MaxOperand {
	typedef MaxOperand this_struct;
	key_t key;
	value_t value;
};

template <typename key_t, typename value_t>
bool operator>(const MaxOperand<key_t, value_t> &a, const MaxOperand<key_t, value_t> &b) {
#pragma HLS INLINE
	return a.key >= b.key; // prefer a
}

template <typename T>
struct LeftOp {
	template <typename U>
	static T Apply(T a, U) {
#pragma HLS INLINE
		return a;
	}
};

template <typename T>
struct RightOp {
	template <typename U>
	static T Apply(U, T b) {
#pragma HLS INLINE
		return b;
	}
};

#endif

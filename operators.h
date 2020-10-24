#ifndef __CNN_HLS_OPERATORS_H__
#define __CNN_HLS_OPERATORS_H__
template <typename key_t, typename value_t>
struct MaxOperand {
	typedef MaxOperand this_struct;
	key_t key;
	value_t value;
	static bool operator>(const this_struct &a, const this_struct &b) {
#pragma HLS INLINE
		return a.key >= b.key; // prefer a
	}
};
#endif

#ifndef __CNN_HLS_MATMUL_H__
#define __CNN_HLS_MATMUL_H__
#include "types.h"
template<typename T, typename OperatorMap, typename OperatorReduce,
	typename RAM_A, typename RAM_B, typename RAM_C>
void matmul_dynamic_ram_naive(RAM_A a, RAM_B b, RAM_C c,
		size_t size_m, size_t size_k, size_t size_n) {
	for (ptrdiff_t i = 0; i < (ptrdiff_t)size_m; i++) {
		for (ptrdiff_t j = 0; j < (ptrdiff_t)size_n; j++) {
			T acc = OperatorReduce::identity();
			for (ptrdiff_t k = 0; k < (ptrdiff_t)size_k; k++) {
				acc = OperatorReduce::Apply(OperatorMap::Apply(
						a[i * size_k + k], b[k * size_m + j]));
			}
			c[i * size_n + j] = acc;
		}
	}
}
#endif

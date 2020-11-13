#include <stddef.h>
#include <stdint.h>
#include <ap_int.h>
#include <ap_fixed.h>
//#include <algorithm>

typedef float value_t;

void naive_matmul_top(const value_t *A, const value_t *B, value_t *C,
		size_t size_m, size_t size_k, size_t size_n) {
#pragma HLS INTERFACE s_axilite port=return

#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=size_m
#pragma HLS INTERFACE s_axilite port=size_k
#pragma HLS INTERFACE s_axilite port=size_n

//	if (size_n > 2) size_n = 2;

	for (ptrdiff_t i = 0; i < (ptrdiff_t)size_m; i++) {
		for (ptrdiff_t j = 0; j < (ptrdiff_t)size_n; j++) {
			value_t sum = 0;
//			C[i * size_n + j] = 0;
			for (ptrdiff_t k = 0; k < (ptrdiff_t)size_k; k++) {
				sum += A[i * size_k + k] * B[k * size_n + j];
//				C[i * size_n + j] += A[i * size_k + k] * B[k * size_n + j];
//				if (sum)
//					C[i * size_n + j] = sum;
			}
			C[i * size_n + j] = sum;
		}
	}
}

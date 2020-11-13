//#include <stddef.h>
//#include <stdint.h>
#include <cstddef>
#include <cstdint>
#include <ap_int.h>
#include <ap_fixed.h>

//#include <sstream>
//#include <hlslib/xilinx/Operators.h>
//#include <hlslib/xilinx/DataPack.h>

void test1_top(ap_int<8> *src, ap_int<8> *dst, size_t len) {
#pragma HLS INTERFACE m_axi port=src offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=dst offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=test1_top bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
	while (len--)
		*dst++ = *src++;
}

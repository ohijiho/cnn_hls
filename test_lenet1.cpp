#include <iostream>
#include <cstring>
#include <algorithm>

typedef float w_t;

#include "./data/weights_conv1.h"
#include "./data/weights_conv2.h"
#include "./data/weights_ip1.h"
#include "./data/weights_ip2.h"
#include "./data/bias_conv1.h"
#include "./data/bias_conv2.h"
#include "./data/bias_ip1.h"
#include "./data/bias_ip2.h"
#include "./data/test_set.h"
#include "./data/label.h"

#include "types.h"

#include "debug.h"

#define N_ITER 1000

static void accuracy(uint32_t iter,											// number of iterations
							uint32_t *label,							// label for test data
							float *output){								// expected outputs
    static int correct_counter = 0;
    uint32_t output_label = 0;
    for (int i = 1; i < 10; i++) {
        if (output[i] > output[output_label])
            output_label = i;
    }
//    return;
//    for (int i = 0; i < 10; i++) {
//        if (i) printf(" ");
//        printf(label[iter] == i ? "[%f]" : output_label == i ? "!!%f!!" : "%f", output[i]);
//    }
//    printf("\n");
    if (label[iter] == output_label) {
        printf("Correct(%d, %d, %f)\n", output_label, label[iter], output[output_label]);
        correct_counter++;
    } else {
        printf("Incorrect(%d, %d, %f, %f)\n", output_label, label[iter], output[output_label], output[label[iter]]);
//        for (ptrdiff_t i = 0; i < 10; i++) {
//        	printf(" %f", output[i]);
//        }
//        printf("\n");
    }
    if (iter == N_ITER - 1)
        printf("Accuracy: %lf\n", (double)correct_counter / (iter + 1));
}

void copy_batch(minibatch_t *dst, const float *src, size_t n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < BATCH_SIZE; j++) {
			dst[i][j] = (value_t)src[j * n + i];
		}
	}
}
void copy_batch(float *dst, const minibatch_t *src, size_t n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < BATCH_SIZE; j++) {
			dst[j * n + i] = (float)src[i][j];
		}
	}
}

#define DEFINE_PARAMETER(name) \
value_t name[sizeof(_##name) / sizeof(float)]
DEFINE_PARAMETER(weights_conv1);
DEFINE_PARAMETER(bias_conv1);
DEFINE_PARAMETER(weights_conv2);
DEFINE_PARAMETER(bias_conv2);
DEFINE_PARAMETER(weights_ip1);
DEFINE_PARAMETER(bias_ip1);
DEFINE_PARAMETER(weights_ip2);
DEFINE_PARAMETER(bias_ip2);
#undef DEFINE_PARAMETER

static void copy_transpose(value_t *dst, const float *src, size_t m, size_t n) {
	for (ptrdiff_t j = 0; j < (ptrdiff_t)n; j++) {
		for (ptrdiff_t i = 0; i < (ptrdiff_t)m; i++) {
			dst[j * m + i] = src[i * n + j];
		}
	}
}

static void prepare_parameters() {
#define COPY_PARAMETER(name, m) \
	copy_transpose(name, _##name, m, sizeof(_##name) / sizeof(float) / (m))
#define COPY_WB_PAIR(name) do { \
		COPY_PARAMETER(weights_##name, sizeof(_bias_##name) / sizeof(float)); \
		COPY_PARAMETER(bias_##name, 1); \
	} while (0)
	COPY_WB_PAIR(conv1);
	COPY_WB_PAIR(conv2);
	COPY_WB_PAIR(ip1);
	COPY_WB_PAIR(ip2);
#undef COPY_PARAMETER
#undef COPY_WB_PAIR
}

static void dump_image(const minibatch_t *img, size_t width, size_t height) {
	for (ptrdiff_t b = 0; b < (ptrdiff_t)BATCH_SIZE; b++) {
		for (ptrdiff_t i = 0; i < (ptrdiff_t)height; i++) {
			for (ptrdiff_t j = 0; j < (ptrdiff_t)width; j++) {
				if (img[i * width + j][b] >= 0.5)
					fputs("#", stdout);
				else
					fputs(" ", stdout);
			}
			printf("\n");
		}
		printf("\n");
	}
	fflush(stdout);
}

float ts_buf[BATCH_SIZE][784];

minibatch_t test_image[2][784];
minibatch_t feature_map1[2][24 * 24 * 5];
minibatch_t max_pool1[2][12 * 12 * 5];
minibatch_t feature_map2[2][8 * 8 * 5];
minibatch_t max_pool2[2][4 * 4 * 5];
minibatch_t ip1[2][40];
minibatch_t tanh1[2][40];
minibatch_t ip2[2][10];

float result[16][10];

int main() {
	const size_t pipe_length = 9;
	printf("(init)\n");fflush(stdout);
//	std::cout << "value_t: " << MaxIdentity<value_t>::value() << std::endl;
//	std::cout << "float: " << MaxIdentity<float>::value() << std::endl;
//	std::cout << "ap_int: " << MaxIdentity<ap_int<32>>::value() << std::endl;
	prepare_parameters();
//	printf("param ok\n");fflush(stdout);
	for (ptrdiff_t i = 0; i < (N_ITER - 1) / BATCH_SIZE + pipe_length; i++) {
		ptrdiff_t bufw = i % 2;
		ptrdiff_t bufr = 1 - bufw;
//		printf("i = %zd\n", i);fflush(stdout);
		if (i < (N_ITER - 1) / BATCH_SIZE + 1) {
			memcpy(ts_buf, ts + 784 * i * BATCH_SIZE,
					std::min<size_t>(BATCH_SIZE, N_ITER - i * BATCH_SIZE) * sizeof(*ts_buf));
			copy_batch(test_image[bufw], ts + 784 * i * BATCH_SIZE, 784);
		}
//		dump_image(test_image[bufw], 28, 28);
//		printf(" input ok\n");fflush(stdout);
		void lenet1(
				const minibatch_t *layer1_x, minibatch_t *layer1_y, const value_t *layer1_weight, const value_t *layer1_bias,
				size_2_t layer1_input_size,
				size_t layer1_in_channels, size_t layer1_out_channels,
				size_2_t layer1_kernel_size, size_2_t layer1_stride, size_2_t layer1_padding, size_2_t layer1_dilation,

				const minibatch_t *layer2_x, minibatch_t *layer2_y,
				size_t layer2_channels, size_2_t layer2_input_size,
				size_2_t layer2_kernel_size, size_2_t layer2_stride, size_2_t layer2_padding, size_2_t layer2_dilation,

				const minibatch_t *layer3_x, minibatch_t *layer3_y, const value_t *layer3_weight, const value_t *layer3_bias,
				size_2_t layer3_input_size,
				size_t layer3_in_channels, size_t layer3_out_channels,
				size_2_t layer3_kernel_size, size_2_t layer3_stride, size_2_t layer3_padding, size_2_t layer3_dilation,

				const minibatch_t *layer4_x, minibatch_t *layer4_y,
				size_t layer4_channels, size_2_t layer4_input_size,
				size_2_t layer4_kernel_size, size_2_t layer4_stride, size_2_t layer4_padding, size_2_t layer4_dilation,

				const minibatch_t *layer5_x, minibatch_t *layer5_y, const value_t *layer5_weight, const value_t *layer5_bias,
				size_t layer5_in_features, size_t layer5_out_features,

				const minibatch_t *layer6_x, minibatch_t *layer6_y, size_t layer6_features,

				const minibatch_t *layer7_x, minibatch_t *layer7_y, const value_t *layer7_weight, const value_t *layer7_bias,
				size_t layer7_in_features, size_t layer7_out_features
				);
		lenet1(
				test_image[bufr], feature_map1[bufw], weights_conv1, bias_conv1,
				{28, 28}, 1, 5, {5, 5}, {1, 1}, {0, 0}, {1, 1},

				feature_map1[bufr], max_pool1[bufw],
				5, {24, 24}, {2, 2}, {2, 2}, {0, 0}, {1, 1},

				max_pool1[bufr], feature_map2[bufw], weights_conv2, bias_conv2,
				{12, 12}, 5, 5, {5, 5}, {1, 1}, {0, 0}, {1, 1},

				feature_map2[bufr], max_pool2[bufw],
				5, {8, 8}, {2, 2}, {2, 2}, {0, 0}, {1, 1},

				max_pool2[bufr], ip1[bufw], weights_ip1, bias_ip1,
				4 * 4 * 5, 40,

				ip1[bufr], tanh1[bufw], 40,

				tanh1[bufr], ip2[bufw], weights_ip2, bias_ip2,
				40, 10
		);
//		printf(" lenet1 done\n");fflush(stdout);

//		{
//			float xbuf[BATCH_SIZE][784];
//			float ybuf[BATCH_SIZE][24 * 24 * 5];
//			copy_batch((float*)xbuf, test_image[bufr], 784);
//			copy_batch((float*)ybuf, feature_map1[bufw], 24 * 24 * 5);
//			for (ptrdiff_t j = 0, iter = (i - 1) * BATCH_SIZE; j < BATCH_SIZE && iter < N_ITER; j++, iter++) {
//                dump_conv(iter, xbuf[j], {28, 28}, 1,
//                        _weights_conv1,
//                        _bias_conv1, 5,
//                        ybuf[j],
//                        {5, 5}, 0, 1);
//			}
//		}
//		{
//			float xbuf[BATCH_SIZE][12 * 12 * 5];
//			float ybuf[BATCH_SIZE][8 * 8 * 5];
//			copy_batch((float*)xbuf, max_pool1[bufr], 12 * 12 * 5);
//			copy_batch((float*)ybuf, feature_map2[bufw], 8 * 8 * 5);
//			for (ptrdiff_t j = 0, iter = (i - 3) * BATCH_SIZE; j < BATCH_SIZE && iter < N_ITER; j++, iter++) {
//				dump_conv(iter, xbuf[j], {12, 12}, 5,
//						_weights_conv2,
//						_bias_conv2, 5,
//						ybuf[j],
//						{5, 5}, 0, 1);
//			}
//		}
//		{
//			float xbuf[BATCH_SIZE][40];
//			float ybuf[BATCH_SIZE][40];
//			copy_batch((float*)xbuf, ip1[bufr], 40);
//			copy_batch((float*)ybuf, tanh1[bufw], 40);
//			for (ptrdiff_t j = 0, iter = (i - 6) * BATCH_SIZE; j < BATCH_SIZE && iter < N_ITER; j++, iter++) {
//				std::cout << "iter " << iter << std::endl;
//				dumpN(std::cout, xbuf[j], 40, ", ");
//				std::cout << std::endl;
//				dumpN(std::cout, ybuf[j], 40, ", ");
//				std::cout << std::endl;
//			}
//		}

		copy_batch((float*)result, ip2[bufr], 10);
//		printf(" output ok\n");fflush(stdout);
		if (i >= 8) {
			for (ptrdiff_t j = 0, iter = (i - 8) * BATCH_SIZE; j < BATCH_SIZE && iter < N_ITER; j++, iter++) {
				accuracy(iter, ls, result[j]);
			}
//		} else {
//			for (ptrdiff_t j = 0, iter = (i - 8) * BATCH_SIZE; j < BATCH_SIZE && iter < N_ITER; j++, iter++) {
//		        for (ptrdiff_t k = 0; k < 10; k++) {
//		        	printf(" %f", result[j][k]);
//		        }
//		        printf("\n");
//			}
		}
		fflush(stdout);
	}
	return 0;
}

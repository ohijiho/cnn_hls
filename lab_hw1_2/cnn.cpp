#include <algorithm>
#include <fstream>
#include <iostream>
#include <math.h>

#include "cnn.h"
#include "config.h"

using namespace std;
w_t test_image[1000][1 * 28 * 28];

/*
 * assumes that size pairs indicate (width, height)
 */

void conv(w_t *image,													// input image
          pair<uint32_t, uint32_t> image_size,							// input image size
          uint32_t num_features,										// number of features in input = channel
          w_t *filter,													// convolution filter source
          w_t *bias,													// convolution bias source
          uint32_t num_filters,											// number of output 
          w_t *feature_map,												// output image
          pair<uint32_t, uint32_t> filter_size,							// filter size
          int32_t pad,													// number of padding
          uint32_t stride) {											// number of stride
	typedef uint32_t uint_t;
	typedef int32_t int_t;

//	pair<uint_t, uint_t> output_size = {
//	        (image_size.first + pad * 2 - filter_size.first) / stride + 1,
//	        (image_size.second + pad * 2 - filter_size.second) / stride + 1};
//	for (int_t m = 0; m < num_filters; m++) {
//	    for (int_t h = 0; h < output_size.second; h++) {
//	        for (int_t w = 0; w < output_size.first; w++) {
//	            w_t output = bias[m];
//	            for (int_t c = 0; c < num_features; c++) {
//	                for (int_t p = 0; p < filter_size.second; p++) {
//	                    for (int_t q = 0; q < filter_size.first; q++) {
//	                        output += image[c * image_size.second * image_size.first + (h + p) * image_size.first + (w + q)]
//	                                * filter[m * num_features * filter_size.second * filter_size.first + c * filter_size.second * filter_size.first + p * filter_size.first + q];
//	                    }
//	                }
//	            }
//	            feature_map[m * output_size.second * output_size.first + h * output_size.first + w] = output;
//	        }
//	    }
//	}
//
//	return;

//	pair<uint_t, uint_t> output_size = {
//	        (image_size.first + pad * 2 - filter_size.first) / stride + 1,
//	        (image_size.second + pad * 2 - filter_size.second) / stride + 1};
//    printf("filter: %f\n", filter[1]);
	for (int_t i = 0; i < num_filters; i++) {
        for (int_t ih = -pad; ih <= image_size.second + pad - filter_size.second; ih += stride) {
            for (int_t iw = -pad; iw <= image_size.first + pad - filter_size.first; iw += stride) {
                w_t output = *bias;
                const w_t *input_ptr = image + ih * image_size.first + iw;
                const w_t *filter_ptr = filter;
                int_t khl = std::max(-ih, 0), khu = std::min(image_size.second - ih, filter_size.second);
                int_t kwl = std::max(-iw, 0), kwu = std::min(image_size.first - iw, filter_size.first);
                for (int_t j = 0; j < num_features; j++) {
                    for (int_t kh = khl; kh < khu; kh++) {
                        for (int_t kw = kwl; kw < kwu; kw++) {
                            output += filter_ptr[kh * filter_size.first + kw] * input_ptr[kh * image_size.first + kw];
//                            if (output) printf("conv_output: %f\n", output);
                        }
                    }
                    input_ptr += image_size.second * image_size.first;
                    filter_ptr += filter_size.second * filter_size.first;
                }
                *feature_map++ = output;
//                if (output) printf("conv_output: %f\n", output);
            }
        }
        filter += num_features * filter_size.second * filter_size.first;
        bias++;
    }
//	printf("conv: %f\n", feature_map[-1]);
}

void max_pool(w_t *image,												// input image
							pair<uint32_t, uint32_t> image_size,		// input image size
							uint32_t channel,							// number of features in input image = channel
							pair<uint32_t, uint32_t> max_pool_size,		// pooling size
							uint32_t stride,							// strdie
							w_t *max_pool) {							// output image
    typedef uint32_t uint_t;
    typedef int32_t int_t;

//    for (int_t m = 0; m < channel; m++) {
//        for (int_t h = 0; h < image_size.second / max_pool_size.second; h++) {
//            for (int_t w = 0; w < image_size.first / max_pool_size.first; w++) {
//                w_t output = std::numeric_limits<w_t>::min();
//                for (int_t p = 0; p < max_pool_size.second; p++) {
//                    for (int_t q = 0; q < max_pool_size.first; q++) {
//                        output = std::max(image[
//                                m * image_size.second * image_size.first
//                                + (h * max_pool_size.second + p) * image_size.first
//                                + (w * max_pool_size.first + q)], output);
//                    }
//                }
//                max_pool[m * (image_size.second / max_pool_size.second) * (image_size.first / max_pool_size.first)
//                        + h * (image_size.first / max_pool_size.first) + w] = output;
//            }
//        }
//    }
//    return;

    while (channel--) {
        for (int_t ih = 0; ih <= image_size.second - max_pool_size.second; ih += stride) {
            for (int_t iw = 0; iw <= image_size.first - max_pool_size.first; iw += stride) {
                const w_t *input = image + ih * image_size.first + iw;
                w_t output = std::numeric_limits<w_t>::lowest();
                for (int_t kh = 0; kh < max_pool_size.second; kh++) {
                    for (int_t kw = 0; kw < max_pool_size.first; kw++) {
                        output = std::max(input[kh * image_size.first + kw], output);
                    }
                }
                *max_pool++ = output;
            }
        }
        image += image_size.second * image_size.first;
    }
//    printf("max_pool: %f\n", max_pool[-1]);
}

void ReLu(w_t *image,													// input image
				pair<uint32_t, uint32_t> image_size,					// input image size
				uint32_t num_output,									// number of output feature			
				w_t *output) {											// output
    typedef uint32_t uint_t;
    typedef int32_t int_t;
    for (uint_t size = num_output * image_size.second * image_size.first; size--;) {
        *output++ = std::max(*image++, (w_t)0);
    }
}

void TanH(w_t *image, 													// input image
				pair<uint32_t, uint32_t> image_size,					// input image size
				uint32_t num_output,									// number of output feature
				w_t *output){											// output
    typedef uint32_t uint_t;
    typedef int32_t int_t;
    for (uint_t size = num_output * image_size.second * image_size.first; size--;) {
        *output++ = tanh(*image++);
    }
}

void ip(w_t *input, pair<uint32_t, uint32_t> input_size,				// input image
				uint32_t num_features,									// number of 1 input's features
				w_t *weight,											// weights
				w_t *bias,												// bias
				uint32_t num_output,									// number of output neurons
				w_t *output){											// output
    typedef uint32_t uint_t;
    typedef int32_t int_t;
    while (num_output--) {
        w_t v = *bias++;
        const w_t *cur_input = input;
        for (uint_t i = num_features * input_size.second * input_size.first; i--;) {
            v += *weight++ * *cur_input++;
        }
        *output++ = v;
    }
}

void accuracy(uint32_t iter,											// number of iterations
							uint32_t *label,							// label for test data
							w_t *output){								// expected outputs
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
    }
    if (iter == 999)
        printf("Accuracy: %lf\n", (double)correct_counter / (iter + 1));
}


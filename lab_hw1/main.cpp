#include <iostream>

#include "cnn.h"
#include "config.h"

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

#include "debug.h"

using namespace std;

int main() {
		for (uint32_t iter = 0; iter < 1000; ++iter){
				for (uint32_t i =0 ; i < 784; ++i){
						test_image[iter][i] = ts[784 * iter + i];
				}

				conv(test_image[iter], make_pair(28, 28), 1,
								_weights_conv1,
								_bias_conv1, 5,
								feature_map1,
								make_pair(5, 5), 0, 1);

//                dump_conv(iter, test_image[iter], make_pair(28, 28), 1,
//                        _weights_conv1,
//                        _bias_conv1, 5,
//                        feature_map1,
//                        make_pair(5, 5), 0, 1);

				max_pool(feature_map1, make_pair(24, 24), 5,
								make_pair(2, 2), 2, max_pool1);


				conv(max_pool1, make_pair(12, 12), 5,
								_weights_conv2,
								_bias_conv2, 5,
								feature_map2,
								make_pair(5, 5), 0, 1);

                dump_conv(iter, max_pool1, make_pair(12, 12), 5,
                        _weights_conv2,
                        _bias_conv2, 5,
                        feature_map2,
                        make_pair(5, 5), 0, 1);

				max_pool(feature_map2, make_pair(8, 8), 5,
								make_pair(2, 2), 2, max_pool2);

				ip(max_pool2,
								make_pair(4, 4), 5,
								_weights_ip1,
								_bias_ip1,
								40, ip1);

				TanH(ip1,	make_pair(1, 1), 40, tanh1);

				ip(tanh1,
								make_pair(1, 1), 40,
								_weights_ip2,
								_bias_ip2,
								10, ip2);

//				conv(test_image[iter], make_pair(28, 28), 1,
//								_weights_conv1,
//								_bias_conv1, 5,
//								feature_map1,
//								make_pair(5, 5), 0, 1);
//                ip(test_image[iter], make_pair(28, 28), 1,
//                        _weights_conv1, _bias_conv1, 5 * 24 * 24, feature_map1);
//
//				max_pool(feature_map1, make_pair(24, 24), 5,
//								make_pair(2, 2), 2, max_pool1);
//
//                ip(max_pool1, make_pair(12, 12), 5,
//                        _weights_ip1, _bias_ip1, 40, ip1);
//
//				TanH(ip1,	make_pair(1, 1), 40, tanh1);
//
//				ip(tanh1,
//								make_pair(1, 1), 40,
//								_weights_ip2,
//								_bias_ip2,
//								10, ip2);

                void accuracy(uint32_t iter, uint32_t *label, w_t *output);
				accuracy(iter, ls, ip2);
		}

		return 0;
}

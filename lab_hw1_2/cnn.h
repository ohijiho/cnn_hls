#ifndef CNN_H_
#define CNN_H_

#include <cstdint>
#include <utility>

typedef float w_t;
extern w_t test_image[][784];

void conv(w_t *image,
				  std::pair<uint32_t, uint32_t> image_size,
				  uint32_t num_features,
		      w_t *filter,
					w_t *bias,
					uint32_t num_filters,
					w_t *feature_map,
				  std::pair<uint32_t, uint32_t> filter_size,
				  int32_t pad,
					uint32_t stride);

void max_pool(w_t *feature_map,
							std::pair<uint32_t, uint32_t> image_size,
							uint32_t channel,
						  std::pair<uint32_t, uint32_t> max_pool_size,
							uint32_t stride,
						  w_t *max_pool);

void ReLu(w_t *image,
					std::pair<uint32_t, uint32_t> image_size,
					uint32_t num_output,
					w_t *output);

void TanH(w_t *image,
					std::pair<uint32_t, uint32_t> image_size,
					uint32_t num_output,
					w_t *output);

void ip(w_t *input, std::pair<uint32_t, uint32_t> input_size,
				uint32_t num_features,
				w_t *weight,
				w_t *bias,
				uint32_t num_output,
				w_t *output);

/*
void accuracy(uint32_t iter,
							uint32_t *label,
							w_t *output);
*/
#endif

#include "config.h"

using namespace std;

float feature_map1[24 * 24 * 5];
float max_pool1[12 * 12 * 5];
float feature_map2[8 * 8 * 5];
float max_pool2[4 * 4 * 5];
float ip1[40];
float tanh1[40];
float ip2[10];

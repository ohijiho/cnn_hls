//
// Created by jiho on 20. 9. 20..
//

#ifndef LAB_HW1_DEBUG_H
#define LAB_HW1_DEBUG_H

#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>

template<typename w_t>
size_t dumpN(std::ostream &out, const w_t *image) {
    out << std::fixed;
    out.precision(4);
    out << *image;
    return 1;
}

template<typename w_t, typename... Tail>
size_t dumpN(std::ostream &out, const w_t *image,
		size_t size, std::string delimiter, Tail... tail) {
    size_t j = 0;
    for (ptrdiff_t i = 0; i < (ssize_t)size; i++) {
        if (i) out << delimiter;
        j += dumpN(out, image + j,
        		tail...);
    }
    return j;
}


template<typename w_t>
void dump_conv(uint32_t iter,
        w_t *image,
        std::pair<uint32_t, uint32_t> image_size,
        uint32_t num_features,
        w_t *filter,
        w_t *bias,
        uint32_t num_filters,
        w_t *feature_map,
        std::pair<uint32_t, uint32_t> filter_size,
        int32_t pad,
        uint32_t stride) {
    std::ostringstream namebuf;
    namebuf << "dump";
    mkdir(namebuf.str().c_str(), 0777); // may fail
    namebuf << "/" << getpid();
    mkdir(namebuf.str().c_str(), 0777); // may fail
    namebuf << "/" << iter << ".txt";
    std::ofstream out(namebuf.str());
    if (!out.is_open()) return;
    out << "Image:" << std::endl;
    dumpN(out, image,
    		(size_t)num_features, ",\n\n",
			(size_t)image_size.second, ",\n",
			(size_t)image_size.first, ", ");
    out << std::endl;
    out << "Filter:" << std::endl;
    dumpN(out, filter,
            (size_t)num_filters, ",\n\n" + std::string(16, '#') + "\n\n",
			(size_t)num_features, ",\n\n",
			(size_t)filter_size.second, ",\n",
			(size_t)filter_size.first, ", ");
    out << std::endl;
    out << "Bias:" << std::endl;
    dumpN(out, bias, (size_t)num_filters, ", ");
    out << std::endl;
    out << "Feature map:" << std::endl;
	std::pair<size_t, size_t> output_size = {
	        (image_size.first + pad * 2 - filter_size.first) / stride + 1,
	        (image_size.second + pad * 2 - filter_size.second) / stride + 1};
    dumpN(out, feature_map,
    		(size_t)num_filters, ",\n\n",
			(size_t)output_size.second, ",\n",
			(size_t)output_size.first, ", ");
    out << std::endl;
}

#endif //LAB_HW1_DEBUG_H

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
size_t dumpN(std::ostream &out, const w_t *image,
        const size_t (&size)[0],
        const std::string (&delimiter)[0]) {
    out << std::fixed;
    out.precision(4);
    out << *image;
    return 1;
}

template<typename w_t, size_t N>
size_t dumpN(std::ostream &out, const w_t *image,
        const size_t (&size)[N],
        const std::string (&delimiter)[N]) {
    if (N == 0) {
        out << std::fixed;
        out.precision(6);
        out << *image;
        return 1;
    }
    size_t j = 0;
    for (ptrdiff_t i = 0; i < (ssize_t)size[0]; i++) {
        if (i) out << delimiter[0];
        j += dumpN(out, image + j,
                *(const size_t (*)[N - 1])(size + 1),
                *(const std::string (*)[N - 1])(delimiter + 1));
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
    dumpN(out, image, {num_features, image_size.second, image_size.first},
            {",\n\n", ",\n", ", "});
    out << std::endl;
    out << "Filter:" << std::endl;
    dumpN(out, filter,
            {num_filters, num_features, filter_size.second, filter_size.first},
            {",\n\n" + std::string(16, '#') + "\n\n", ",\n\n", ",\n", ", "});
    out << std::endl;
    out << "Bias:" << std::endl;
    dumpN(out, bias, {num_filters, }, {", "});
    out << std::endl;
    out << "Feature map:" << std::endl;
	std::pair<size_t, size_t> output_size = {
	        (image_size.first + pad * 2 - filter_size.first) / stride + 1,
	        (image_size.second + pad * 2 - filter_size.second) / stride + 1};
    dumpN(out, feature_map, {num_filters, output_size.second, output_size.first}, {",\n\n", ",\n", ", "});
    out << std::endl;
}

#endif //LAB_HW1_DEBUG_H

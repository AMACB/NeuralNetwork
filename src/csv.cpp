/* Copyright 2017 Alexander Burton. All rights reserved */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "csv.h"

namespace csv {
    std::vector<csv::dataset*> load_test_cases(std::string filename) {
        logger(logINFO) << "[CSV] Loading test data...\n";
        std::vector<csv::dataset*> result;
        std::string line;
        std::ifstream in_file(filename);
        if (!in_file.is_open()) {
            logger(logERROR) << "Could not read file " << filename;
            exit(1);
        }
        logger(logINFO) << "[CSV] Reading CSV file...\n";
        logger(logINFO) << "[CSV] Loading test cases...\n";
        int current_count = 0;
        while (std::getline(in_file, line)) {
            current_count++;
            logger(logINFO) << "\r[CSV] Loading test case " << current_count;
            std::stringstream ss(line);
            std::vector<std::string> v;
            std::string field;
            while (std::getline(ss, field, ',')) {
                v.push_back(field);
            }
            csv::dataset* data = new csv::dataset;
            data->input.resize(INPUT_LENGTH);
            data->output.resize(OUTPUT_LENGTH);
            for (size_t i = 0; i < v.size() && i < data->input.size() + 1; ++i) {
                if (i == 0) {
                    std::fill(data->output.begin(), data->output.end(), false);
                    size_t num = std::stoul(v[0]);
                    if (!(num < OUTPUT_LENGTH)) {
                        std::cerr << "Number out of range" << std::endl;
                        throw 1;
                    }
                    data->output.at(num) = true;
                    // data.output = num;
                } else {
                    data->input[i-1] = static_cast<double>(std::stoi(v[i]))/255.0;
                }
            }
            result.push_back(data);
        }
        in_file.close();
        logger(logINFO) << "\r[CSV] Test data loaded       \n";
        return result;
    }
}  // namespace csv

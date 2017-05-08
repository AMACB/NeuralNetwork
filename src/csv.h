/* Copyright 2017 Alexander Burton. All rights reserved */

#ifndef SRC_CSV_H_
#define SRC_CSV_H_

#include <iostream>
#include <string>
#include <vector>

#include "log/logging.h"

#define INPUT_LENGTH 784
#define OUTPUT_LENGTH 10

namespace csv {
struct dataset {
    std::vector<double> input;
    std::vector<bool> output;
};
std::vector<csv::dataset*> load_test_cases(std::string);
}

#endif  // SRC_CSV_H_

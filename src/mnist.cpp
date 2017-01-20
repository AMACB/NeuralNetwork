#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "mnist.hpp"

namespace mnist {
	std::vector<mnist::dataset*> load_test_cases(std::string filename) {
		std::cout << "[MNIST] Loading test data..." << std::endl;
		std::vector<mnist::dataset*> result;
		std::string line;
		std::ifstream in_file(filename);
		if (!in_file.is_open()) {
			std::cerr << "Could not read file " << filename << std::endl;
		}
		std::cout << "[MNIST] Reading CSV file..." << std::endl;
		std::cout << "[MNIST] Loading test cases..." << std::endl;
		int current_count = 0;
		while (std::getline(in_file, line)) {
			current_count++;
			std::cout << "\r[MNIST] Loading test case " << current_count << std::flush;
			std::stringstream ss(line);
			std::vector<std::string> v;
			std::string field;
			while (std::getline(ss, field, ',')) {
				v.push_back(field);
			}
			mnist::dataset* data = new mnist::dataset;
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
					data->input[i-1] = double(std::stoi(v[i]))/255.0;
				}
			}
			result.push_back(data);
		}
		in_file.close();
		std::cout << "\r[MNIST] Test data loaded       " << std::endl;
		return result;
	}
}

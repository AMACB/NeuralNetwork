#ifndef NEURAL_NETWORK_MNIST_H_
#define NEURAL_NETWORK_MNIST_H_

#include <iostream>
#include <vector>

#define INPUT_LENGTH 784
#define OUTPUT_LENGTH 10

namespace mnist {
	struct dataset {
		std::vector<double> input;
		std::vector<bool> output;
	};

	std::vector<mnist::dataset*> load_test_cases(std::string);
}

#endif // NEURAL_NETWORK_MNIST_H_

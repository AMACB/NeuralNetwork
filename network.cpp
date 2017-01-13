#include <algorithm>
#include <array>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "mnist.cpp"
#include "matrix.cpp"

typedef std::chrono::steady_clock::time_point time_point;
#define now() std::chrono::steady_clock::now();

class Network {
public:
	typedef double ddouble;
	typedef std::vector<ddouble> double_v;

	int num_layers;
	std::vector<int> sizes;

/* 
 * Vector of all layers, length = number of layers
 * Each layer is a vector of Neurons, length = number of neuron in layer
 *  aka a Matrix
 * Each Neuron is a vector of biases or weight
 *  aka a Matrix
 * 	For weights, length = number of neuron in previous layer
 * 	For biases, length = 1
 */
	std::vector<Matrix*> biases;
	std::vector<Matrix*> weights;

	/* Debug function for printing a vector */
	/*
	template <typename T> static void print_v(std::vector<T> v) {
		std::cout << "[";
		for (int i = 0; i < v.size(); ++i) {
			std::cout << v[i] << ",";
		}
		std::cout << "]";
	}
	*/

	/* 
     * Returns num_samples samples of double_v of length
     * len_samples of normal distribution
     */
	static Matrix normal_samples(int num_samples, int len_samples) {
		std::random_device rd;
		std::default_random_engine generator(rd());
		std::normal_distribution<double> normal_dist;
		Matrix samples(num_samples, len_samples);
		for (int i = 0; i < num_samples; ++i) {
			for (int j = 0; j < len_samples; ++j) {
				samples.set_index(i,j,(normal_dist(generator)));
			}
		}
		return samples;
	}

	/* Calculates the sigmoid (prime) function of a real number */
	static ddouble sigmoid(ddouble a) {
		return 1/(1+std::exp(-a));
	}

	static ddouble sigmoid_prime(ddouble a) {
		return Network::sigmoid(a) * (1 - Network::sigmoid(a));
	}

	/* Calculates the element-wise sigmoid (prime) function of a vector */
	static double_v sigmoid(double_v a) {
		double_v result;
		for (int i = 0; i < a.size(); ++i) {
			result.push_back(sigmoid(double(a[i])));
		}
		return result;
	}

	static double_v sigmoid_prime(double_v a) {
		double_v result;
		for (int i = 0; i < a.size(); ++i) {
			result.push_back(sigmoid_prime(double(a[i])));
		}
		return result;
	}

	/* Calculates the element-wise sigmoid(prime) function of a matrix */
	static Matrix sigmoid(Matrix a) {
		for (int i = 0; i < a.rows; ++i) {
			for (int j = 0; j < a.cols; ++j) {
				a.set_index(i,j,(sigmoid(a.index(i,j))));
			}
		}
		return a;
	}

/*
 * Initializes the Network give the sizes, where 
 * the nth element is the number of neurons in the nth layer
 */
	Network(std::vector<int> sizes) {
		std::vector<int>::iterator iter;
		/*
		 * Initialize biases
		 * Biases have length one because each neuron has a single bias
		 */
		int i = 0;
		for (iter = sizes.begin() + 1; iter < sizes.end(); ++i, ++iter) {
			/* Pushes a single layer at a time */
			Matrix* samples = new Matrix;
			*samples = Network::normal_samples(*iter, 1);
			biases.push_back(samples);
		}
		std::vector<int>::iterator iter_2;
		/*
		 * Initialize weights
		 * Weights have length of previous 
		 */
		iter = sizes.begin();
		for (iter_2 = iter + 1; iter_2 < sizes.end(); ++iter, ++iter_2) {
			/* iter is one element behind iter_2 */
			/* Pushes a single layer at a time */
			Matrix* samples = new Matrix;
			*samples = Network::normal_samples(*iter_2, *iter);
			weights.push_back(samples);
		}

		/*
		 * Set number of layers
		 */
		this->num_layers = sizes.size();
	}

	~Network() {
		for (int i = 0; i < this->weights.size(); ++i) {
			delete this->weights.at(i);
		}
		for (int i = 0; i < this->biases.size(); ++i) {
			delete this->biases.at(i);
		}
	}

	double_v feedforward(double_v current) {
		double_v activation = current;
		Matrix activation_mat = Matrix(activation).transposed();
		/* Iterate through each Matrix of both biases and weights */
		for (int i = 0; i < (this->biases).size() && i < (this->weights).size(); ++i) {
			Matrix *bias = this->biases[i], *weights = this->weights[i];
			activation_mat = Matrix(
				Network::sigmoid((
					(*weights * activation_mat) + bias->flatten()
				).flatten())
			).transposed();
		}
		return activation_mat.flatten();
	}

	/* Stochastic Gradient Decent Algorithm */
	void SGD(std::vector<mnist::dataset*>* training_data, int num_epochs,
	 		int mini_batch_size, double learning_rate, std::vector<mnist::dataset*>* test_data) {
		/* Initialize variables with useful measures */
		std::cout << "[SGD] Beginning SGD..." << std::endl;
		std::cout << "[SGD] Number of epochs: " << num_epochs << std::endl;
		std::cout << "[SGD] Mini batch size: " << mini_batch_size << std::endl;
		std::cout << "[SGD] Learning rate: " << learning_rate << std::endl;

		int n_test = 0; // number of test datasets
		if (test_data->size() > 0) n_test = test_data->size();
		std::cout << "[SGD] Number of test datasets: " << n_test << std::endl;
		int n = training_data->size(); // number of training datasets
		std::cout << "[SGD] Number of training datasets: " << n << std::endl;
		int num_mini_batches = n / mini_batch_size; // the number of minibatches
		std::cout << "[SGD] Number of minibatches: " << num_mini_batches << std::endl;
		int num_leftover = n % mini_batch_size;
		std::cout << "[SGD] Number of leftover datasets: " << num_leftover << std::endl;
		/* n mod mini_batch_size datasets are excluded to make sure minibatches have same size */
		
		for (int j = 0; j < num_epochs; ++j) {
			std::cout << "[SGD] Beginning epoch " << j+1 << std::endl;
			std::random_shuffle(training_data->begin(), training_data->end());
			std::vector<std::vector<mnist::dataset*> > mini_batches(num_mini_batches, std::vector<mnist::dataset*>(mini_batch_size)); // vector of all minibatches
			/* Create minibatches with random elements */
			int minibatch_num = 0;
			for (int k = 0; k < n - num_leftover; k += mini_batch_size, ++minibatch_num) {
				for (int l = 0; l < mini_batch_size; ++l) {
					mini_batches[minibatch_num][l] = (*training_data)[k + l];
				}
			}

			std::cout << "[SGD] Updating minibatches..." << std::endl;
			std::cout << "[SGD] Updating minibatch 0";
			/* Update each minibatch */
			for (int i = 0; i < num_mini_batches; ++i) {
				std::cout << "\r[SGD] Updating minibatch " << i+1 << std::flush;
				this->update_mini_batch(&mini_batches[i], learning_rate);
			}
			std::cout << "\r[SGD] Done updating minibatches" << std::endl;

			if (test_data->size() > 0) {
				std::cout << "[SGD] Evaluating test dataset..." << std::endl;
				std::cout << "Epoch " << j+1 << ": " << this->evaluate(test_data) << " / " << n_test << std::endl;
			} else {
				std::cout << "Epoch " << j+1 << " complete." << std::endl;
			}
		}
	}

	/* Updates given a mini batch of datasets */
	void update_mini_batch(std::vector<mnist::dataset*>* mini_batch, double learning_rate) {
		std::vector<Matrix> nabla_b, nabla_w;
		
		/* Create zero-filled matrix of same dimensions */
		for (int i = 0; i < this->biases.size(); ++i) {
			Matrix bias = Matrix(*this->biases[i]); // copy
			bias.zeroify();
			nabla_b.push_back(bias);
		}
		for (int i = 0; i < this->weights.size(); ++i) {
			Matrix weight = Matrix(*this->weights[i]); // copy
			weight.zeroify();
			nabla_w.push_back(weight);
		}

		/* Find the deltas with back propogation */
		for (int i = 0; i < mini_batch->size(); ++i) {
			mnist::dataset* dataset = mini_batch->at(i);

			/* Use backpropogation to find the deltas */
			// The std::pair<...> storage class is extremely slow, resulting in this code being slow as well
			std::pair<std::vector<Matrix*>, std::vector<Matrix*> >* adjusted = this->backprop(dataset);

			for (int i = 0; i < nabla_b.size() && i < adjusted->first.size(); ++i) {
				nabla_b[i] += *adjusted->first[i];
				delete adjusted->first[i];
			}
			for (int i = 0; i < nabla_w.size() && i < adjusted->second.size(); ++i) {
				nabla_w[i] += *adjusted->second[i];
				delete adjusted->second[i];
			}
			delete adjusted;
			
			// std::cout << "took " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " ms" << std::endl;
		}
		/* Update the biases and weights */
		for (int i = 0; i < this->biases.size() && i < nabla_b.size(); ++i) {
			Matrix nabla_b_val = nabla_b[i] * (- learning_rate / mini_batch->size());
			*this->biases[i] += nabla_b_val;
		}
		for (int i = 0; i < this->weights.size() && i < nabla_w.size(); ++i) {
			Matrix nabla_w_val = nabla_w[i] * (- learning_rate / mini_batch->size());
			*this->weights[i] += nabla_w_val;
		}
	}

	/* Uses the backpropogation algorithm to calculate the errors */
	/* Returns a std::pair with delta nabla of biases; and delta nabla of weights */
	std::pair<std::vector<Matrix*>, std::vector<Matrix*> >* backprop(mnist::dataset* dataset) {
		std::vector<Matrix*> nabla_w(this->weights.size()), nabla_b(this->biases.size());

		/* Feedforward and save activations and z values into vector of vectors */
		std::vector<double_v> activations, zvalues;
		double_v activation(dataset->input.begin(), dataset->input.end());
		activations.push_back(activation);

		Matrix activation_mat = Matrix(activation).transposed();
		
		for (int i = 0; i < this->biases.size() && i < this->weights.size(); ++i) {
			Matrix *bias = this->biases[i], *weight = this->weights[i];
			double_v zvalue = ((*weight * activation_mat) + bias->flatten()).flatten();
			zvalues.push_back(zvalue);

			activation = Network::sigmoid(zvalue);
			activation_mat = Matrix(activation).transposed();
			activations.push_back(activation);
		}

		/* Backward pass */
		double_v delta = this->cost_derivative(activations.back(), dataset->output);
		for (int i = 0; i < delta.size() && i < zvalues.back().size(); ++i) {
			delta.at(i) *=  Network::sigmoid_prime(zvalues.back().at(i));
		}

		/* Calculate the last layer of deltas */
		Matrix delta_mat(delta);
		delta_mat = delta_mat.transposed();
		nabla_b.back() = new Matrix(delta_mat);
		Matrix activations_sec = Matrix(activations.at(activations.size() - 2));
		nabla_w.back() = new Matrix(delta_mat * activations_sec);

		double_v z_val, sp;

		/* Iterate through layers to find deltas based on previous */
		for (int l = 2; l < this->num_layers; ++l) {
			z_val = zvalues.at(zvalues.size() - l);
			sp = Network::sigmoid_prime(z_val);

			Matrix weights_transposed = Matrix(this->weights.at(this->weights.size() - l+1)->transposed());
			delta_mat = Matrix::inner_product((weights_transposed * delta_mat), Matrix(sp).transposed());

			Matrix activations_mat = Matrix(activations.at(activations.size() - l-1));
			nabla_b[nabla_b.size() - l] = new Matrix(delta_mat);
			nabla_w[nabla_w.size() - l] = new Matrix(delta_mat * activations_mat);
		}
		
		// std::cout << "took " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " ms" << std::endl;
		
		std::pair<std::vector<Matrix*>, std::vector<Matrix*> >* pair = new std::pair<std::vector<Matrix*>, std::vector<Matrix*> >;
		pair->first = nabla_b;
		pair->second = nabla_w;
		return pair;
	}

	/* Find the difference between the correct output and the final activations */
	double_v cost_derivative(const double_v& activation, const std::vector<bool>& output) {
		if (activation.size() != output.size()) {
			std::cerr << "Error! Final activation different length than correct output" << std::endl;
		}
		double_v result;
		double_v::const_iterator iter_act = activation.begin();
		std::vector<bool>::const_iterator iter_out = output.begin();
		for (; iter_act < activation.end() && iter_out < output.end(); ++iter_act, ++iter_out) {
			result.push_back(*iter_act - *iter_out);
		}
		return result;
	}

	/* Evaluates a set of data and returns the number of sucessful datasets */
	/* May have error */
	int evaluate(std::vector<mnist::dataset*>* datasets) {
		int num_worked = 0;
		for (int i = 0; i < datasets->size(); ++i) {
			mnist::dataset* dataset = (*datasets)[i];
			double_v inputs;
			for (int i = 0; i < dataset->input.size(); ++i) {
				inputs.push_back(dataset->input.at(i));
			}

			/* Get result for the input */
			double_v result = this->feedforward(inputs);
			int final_result = this->get_result(result);

			/* Compare the output from the network with the correct result */
			if (dataset->output[final_result]) num_worked++;
		}
		return num_worked;
	}

	/* 
     * Converts the output of a neural network to a boolean array, 
     * where the only truthy element is the greatest from the double_v.
     */
	static int get_result(const double_v& output) {
		double_v::const_iterator iter = std::max_element(output.begin(), output.end());
		int index_of_max = std::distance(output.begin(), iter);
		return index_of_max;
	}
};

int main() {
	std::vector<int> sizes;
	sizes.push_back(784); sizes.push_back(30); sizes.push_back(10);
	Network network(sizes);

	std::vector<mnist::dataset*> test_cases = mnist::load_test_cases("mnist_test.csv");
	//std::vector<mnist::dataset*> train_cases = mnist::load_test_cases("mnist_train.csv");
	network.SGD(&test_cases, 30, 10, 3.0, &test_cases);
}

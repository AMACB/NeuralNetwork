#ifndef NEURAL_NETWORK_NETWORK_H_
#define NEURAL_NETWORK_NETWORK_H_

#include "matrix.hpp"
#include "mnist.hpp"

typedef double ddouble;
typedef std::vector<ddouble> double_v;

class Network {
public:
	size_t num_layers;
	std::vector<size_t> sizes;

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

	/* 
     * Returns num_samples samples of double_v of length
     * len_samples of normal distribution
     */
	static Matrix* normal_samples(size_t, size_t);

	/* Calculates the sigmoid (prime) function of a real number */
	static ddouble sigmoid(ddouble);

	static ddouble sigmoid_prime(ddouble);

	/* Calculates the element-wise sigmoid (prime) function of a vector */
	static double_v sigmoid(double_v a);

	static double_v sigmoid_prime(double_v);

	/* Calculates the element-wise sigmoid(prime) function of a matrix */
	static Matrix sigmoid(Matrix);

/*
 * Initializes the Network give the sizes, where 
 * the nth element is the number of neurons in the nth layer
 */
	Network(std::vector<size_t>);
	
/*
 * Loads the network from a previously-saved file
 */
	Network(std::string);

	~Network();

	double_v feedforward(double_v current);

	/* Stochastic Gradient Decent Algorithm */
	void SGD(std::vector<mnist::dataset*>*, std::vector<mnist::dataset*>*,
			size_t, size_t, double);

	/* Updates given a mini batch of datasets */
	void update_mini_batch(std::vector<mnist::dataset*>*, double);

	/* Uses the backpropogation algorithm to calculate the errors */
	/* Returns a std::pair with delta nabla of biases; and delta nabla of weights */
	std::pair<std::vector<Matrix>, std::vector<Matrix> >* backprop(mnist::dataset* dataset);

	/* Find the difference between the correct output and the final activations */
	double_v cost_derivative(const double_v& activation, const std::vector<bool>& output);

	/* Evaluates a set of data and returns the number of sucessful datasets */
	int evaluate(std::vector<mnist::dataset*>* datasets);
	
	/* 
     * Converts the output of a neural network to a boolean array, 
     * where the only truthy element is the greatest from the double_v.
     */
	static int get_result(const double_v& output);
};


#endif  // NEURAL_NETWORK_NETWORK_H_

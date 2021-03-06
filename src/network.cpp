/* Copyright 2017 Alexander Burton. All rights reserved. */

// TODO(amacburton@gmail.com)
// - choice of initializer
// - choice of sigmoid function
// - choice of cross entropy function

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "network.h"

typedef std::chrono::steady_clock::time_point time_point;
#define now() std::chrono::steady_clock::now();

namespace network {

/*
 * Returns num_samples samples of double_v of length
 * len_samples of normal distribution
 */
Matrix* Network::normal_samples(size_t num_samples, size_t len_samples) {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> normal_dist;
    Matrix* samples = new Matrix(num_samples, len_samples);
    for (double_v::iterator iter = samples->data.begin(); iter != samples->data.end(); ++iter) {
        *iter = (normal_dist(generator));
    }
    return samples;
}

/* Calculates the sigmoid (prime) function of a real number */
inline ddouble Network::sigmoid(ddouble a) {
    return 1/(1+std::exp(-a));
}

inline ddouble Network::sigmoid_prime(ddouble a) {
    return Network::sigmoid(a) * (1 - Network::sigmoid(a));
}

/* Calculates the element-wise sigmoid (prime) function of a vector */
double_v Network::sigmoid(double_v a) {
    double_v result;
    for (double_v::const_iterator iter = a.begin(); iter != a.end(); ++iter) {
        result.push_back(sigmoid(static_cast<double>(*iter)));
    }
    return result;
}

double_v Network::sigmoid_prime(double_v a) {
    double_v result;
    for (double_v::const_iterator iter = a.begin(); iter != a.end(); ++iter) {
        result.push_back(sigmoid_prime(static_cast<double>(*iter)));
    }
    return result;
}

/* Calculates the element-wise sigmoid(prime) function of a matrix */
Matrix Network::sigmoid(Matrix a) {
    for (double_v::iterator iter = a.data.begin(); iter != a.data.end(); ++iter) {
        *iter = sigmoid(*iter);
    }
    return a;
}

/*
* Initializes the Network give the sizes, where 
* the nth element is the number of neurons in the nth layer
* Uses Gaussian distribution to generate weights and biases divided by sqrt(connected neurons)
*/
Network::Network(std::vector<size_t> input_sizes) {
    std::vector<size_t>::iterator iter;
    /*
     * Initialize biases
     * Biases have length one because each neuron has a single bias
     */
    size_t i = 0;
    for (iter = input_sizes.begin() + 1; iter < input_sizes.end(); ++i, ++iter) {
        /* Pushes a single layer at a time */
        Matrix* samples = Network::normal_samples(*iter, 1);
        biases.push_back(samples);
    }
    std::vector<size_t>::iterator iter_2;
    /*
     * Initialize weights
     * Weights have length of previous 
     */
    iter = input_sizes.begin();
    for (iter_2 = iter + 1; iter_2 < input_sizes.end(); ++iter, ++iter_2) {
        /* iter is one element behind iter_2 */
        /* Pushes a single layer at a time */
        Matrix* samples = Network::normal_samples(*iter_2, *iter);
        *samples *= (1.0 / std::sqrt(*iter));
        weights.push_back(samples);
    }
    /*
     * Set number of layers
     */
    this->num_layers = input_sizes.size();
}

/*
* Loads the network from a previously-saved file
*/
Network::Network(std::string filename) {
    std::ifstream infile;
    infile.open(filename);
    if (!infile.is_open()) {
        logger(logERROR) << "Error reading file " << filename;
        exit(1);
    }
    std::string line;
    
    enum {
        BIASES, WEIGHTS        
    } mode = BIASES;
    
    while (infile.good()) {
        std::getline(infile, line);
        if (line == "biases") {
            mode = BIASES;
        } else if (line == "weights") {
            mode = WEIGHTS;
        } else if (line == "end") {
            break;
        } else {
            Matrix* mat = new Matrix(line);
            if (mode == BIASES) {
                this->biases.push_back(mat);
            } else if (mode == WEIGHTS) {
                this->weights.push_back(mat);
            }
        }
    }
    if (this->biases.size() != this->weights.size()) {
        throw std::runtime_error("Error in loading matrix: biases different size than weights");
    }
    this->num_layers = this->biases.size() + 1;
    std::vector<size_t> _sizes;
    _sizes.push_back(this->weights.at(0)->cols);
    for (Matrix* m : this->weights) {
        _sizes.push_back(m->rows);
    }
    infile.close();
}

Network::~Network() {
    for (auto iter = this->weights.begin(); iter != this->weights.end(); ++iter) {
        delete *iter;
    }
    for (auto iter = this->biases.begin(); iter != this->biases.end(); ++iter) {
        delete *iter;
    }
}

double_v Network::feedforward(double_v current) {
    double_v activation = current;
    Matrix activation_mat = Matrix(activation).transposed();
    /* Iterate through each Matrix of both biases and weights */
    for (size_t i = 0; i < (this->biases).size() && i < (this->weights).size(); ++i) {
        Matrix *bias = this->biases[i], *weight = this->weights[i];
        activation_mat = Matrix(
            Network::sigmoid((
                (*weight * activation_mat) + bias->flatten())
            .flatten()))
        .transposed();
    }
    return activation_mat.flatten();
}

/* Stochastic Gradient Decent Algorithm */
void Network::SGD(std::vector<mnist::dataset*>* training_data, std::vector<mnist::dataset*>* test_data,
        size_t num_epochs, size_t mini_batch_size, double learning_rate, double lambda) {
    /* Initialize variables with useful measures */
    logger(logPROGRESS) << "[SGD] Beginning SGD..." << "\n";
    logger(logINFO) << "[SGD] Number of epochs: " << num_epochs << "\n";
    logger(logINFO) << "[SGD] Mini batch size: " << mini_batch_size << "\n";
    logger(logINFO) << "[SGD] Learning rate: " << learning_rate << "\n";

    size_t n_test           = test_data->size();       // number of test datasets
    size_t n                = training_data->size();   // number of training datasets
    size_t num_mini_batches = n / mini_batch_size;     // number of minibatches
    size_t num_leftover     = n % mini_batch_size;     // number of leftover minibatches

    /* n mod mini_batch_size datasets are excluded to make sure minibatches have same size */

    for (size_t j = 0; j < num_epochs; ++j) {
        logger(logINFO) << "[SGD] Beginning epoch " << j+1 << '\n';
        std::random_shuffle(training_data->begin(), training_data->end());
        /* Vector of all minibatches */
        std::vector<std::vector<mnist::dataset*> > mini_batches(num_mini_batches,
            std::vector<mnist::dataset*>(mini_batch_size));
        /* Create minibatches with random elements */
        size_t minibatch_num = 0;
        for (size_t k = 0; k < n - num_leftover; k += mini_batch_size, ++minibatch_num) {
            for (size_t l = 0; l < mini_batch_size; ++l) {
                mini_batches[minibatch_num][l] = (*training_data)[k + l];
            }
        }

        logger(logINFO) << "[SGD] Updating minibatches..." << '\n';
        logger(logINFO) << "[SGD] Updating minibatch 0";
        /* Update each minibatch */
        for (size_t i = 0; i < num_mini_batches; ++i) {
            logger(logINFO) << "\r[SGD] Updating minibatch " << i+1;
            this->update_mini_batch(&mini_batches[i], learning_rate, lambda, n);
        }
        logger(logINFO) << "\r[SGD] Done updating minibatches\n";

        if (test_data->size() > 0) {
            logger(logINFO) << "[SGD] Evaluating test dataset...\n";
            logger(logPROGRESS) << "Epoch " << j+1 << ": "
                << this->evaluate(test_data) << " / " << n_test << "\n";
        } else {
            logger(logPROGRESS) << "Epoch " << j+1 << " complete." << "\n";
        }
    }
}

/* Updates given a mini batch of datasets */
void Network::update_mini_batch(std::vector<mnist::dataset*>* mini_batch,
        double learning_rate, double lambda, int training_data_size) {
    std::vector<Matrix> nabla_b, nabla_w;

    /* Create zero-filled matrix of same dimensions */
    for (size_t i = 0; i < this->biases.size(); ++i) {
        Matrix bias = Matrix(*this->biases[i]);
        bias.zeroify();
        nabla_b.push_back(bias);
    }
    for (size_t i = 0; i < this->weights.size(); ++i) {
        Matrix weight = Matrix(*this->weights[i]);
        weight.zeroify();
        nabla_w.push_back(weight);
    }

    /* Find the deltas with back propogation */
    for (size_t i = 0; i < mini_batch->size(); ++i) {
        mnist::dataset* dataset = mini_batch->at(i);

        /* Use backpropogation to find the deltas */
        // The std::pair<...> storage class is extremely slow, resulting in this code being slow as well
        std::pair<std::vector<Matrix>, std::vector<Matrix> >* adjusted = this->backprop(dataset);

        for (size_t j = 0; j < nabla_b.size() && j < adjusted->first.size(); ++j) {
            nabla_b[j] += (adjusted->first[j]);
            // delete adjusted->first[j];
        }
        for (size_t j = 0; j < nabla_w.size() && j < adjusted->second.size(); ++j) {
            nabla_w[j] += (adjusted->second[j]);
            // delete adjusted->second[j];
        }
        delete adjusted;
        // std::cout << "took " << std::chrono::duration_cast<std::chrono::microseconds>
        //     (end - begin).count() / 1000.0 << " ms" << std::endl;
    }
    /* Update the biases and weights */
    for (size_t i = 0; i < this->biases.size() && i < nabla_b.size(); ++i) {
        *this->biases[i] *= (1 - learning_rate * (lambda / training_data_size));
        *this->biases[i] += (nabla_b[i] * (- learning_rate / mini_batch->size()));
    }
    for (size_t i = 0; i < this->weights.size() && i < nabla_w.size(); ++i) {
        *this->weights[i] += (nabla_w[i] * (- learning_rate / mini_batch->size()));
    }
}

/* Uses the backpropogation algorithm to calculate the errors */
/* Returns a std::pair with delta nabla of biases; and delta nabla of weights */
std::pair<std::vector<Matrix>, std::vector<Matrix> >* Network::backprop(mnist::dataset* dataset) {
    std::vector<Matrix> nabla_w(this->weights.size()), nabla_b(this->biases.size());

    /* Feedforward and save activations and z values into vector of vectors */
    std::vector<double_v> activations, zvalues;
    double_v activation(dataset->input.begin(), dataset->input.end());
    activations.push_back(activation);
    Matrix activation_mat = Matrix(activation).transposed();
    for (size_t i = 0; i < this->biases.size() && i < this->weights.size(); ++i) {
        Matrix *bias = this->biases[i], *weight = this->weights[i];
        double_v zvalue = ((*weight * activation_mat) + bias->flatten()).flatten();
        zvalues.push_back(zvalue);

        activation = Network::sigmoid(zvalue);
        activation_mat = Matrix(activation).transposed();
        activations.push_back(activation);
    }

    /* Backward pass */
    // double_v delta = this->cost_derivative(activations.back(), dataset->output);
    double_v delta = Network::cross_entropy_delta(zvalues.back(), activations.back(),
        std::vector<double>(dataset->output.begin(), dataset->output.end()));

    for (size_t i = 0; i < delta.size() && i < zvalues.back().size(); ++i) {
        delta.at(i) *=  Network::sigmoid_prime(zvalues.back().at(i));
    }

    /* Calculate the last layer of deltas */
    Matrix delta_mat(delta);
    delta_mat = delta_mat.transposed();
    nabla_b.back() = Matrix(delta_mat);
    Matrix activations_sec = Matrix(activations.at(activations.size() - 2));
    nabla_w.back() = Matrix(delta_mat * activations_sec);
    double_v z_val, sp;
    /* Iterate through layers to find deltas based on previous */
    for (size_t l = 2; l < this->num_layers; ++l) {
        z_val = zvalues.at(zvalues.size() - l);
        sp = Network::sigmoid_prime(z_val);

        Matrix weights_transposed = Matrix(this->weights.at(this->weights.size() - l+1)->transposed());
        delta_mat = Matrix::inner_product((weights_transposed * delta_mat), Matrix(sp).transposed());

        Matrix activations_mat = Matrix(activations.at(activations.size() - l-1));
        nabla_b.at(nabla_b.size() - l) = Matrix(delta_mat);
        nabla_w.at(nabla_w.size() - l) = Matrix(delta_mat * activations_mat);
    }

    auto pair = new std::pair<std::vector<Matrix>, std::vector<Matrix> >;
    pair->first = nabla_b;
    pair->second = nabla_w;
    return pair;
}

/* Find the difference between the correct output and the final activations */
double_v Network::cost_derivative(const double_v& activation, const std::vector<bool>& output) {
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
int Network::evaluate(std::vector<mnist::dataset*>* datasets) {
    size_t num_worked = 0;
    for (size_t i = 0; i < datasets->size(); ++i) {
        mnist::dataset* dataset = (*datasets)[i];
        double_v inputs;
        for (size_t j = 0; j < dataset->input.size(); ++j) {
            inputs.push_back(dataset->input.at(j));
        }

        /* Get result for the input */
        double_v result = this->feedforward(inputs);
        int final_result = this->get_result(result);

        /* Compare the output from the network with the correct result */
        if (dataset->output[static_cast<size_t>(final_result)]) num_worked++;
    }
    return num_worked;
}

/* 
 * Converts the output of a neural network to a boolean array, 
 * where the only truthy element is the greatest from the double_v.
 */
int Network::get_result(const double_v& output) {
    double_v::const_iterator iter = std::max_element(output.begin(), output.end());
    int index_of_max = std::distance(output.begin(), iter);
    return index_of_max;
}

/*
 * Quadratic cost function
 * Returns the cost with actual output and desired output
 */
double Network::quadratic_cost(const double_v& actual, const double_v& desired) {
    if (actual.size() == desired.size()) {
        double total = 0;
        for (double_v::const_iterator a = actual.begin(), d = desired.begin();
                a != actual.end() && d != desired.end(); ++a, ++d) {
            double val = *a - *d;
            total += val * val;
        }
        return 0.5 * total;
    } else {
        logger(logERROR) << "Error in computing quadratic cost: "
            << actual.size() << " and " << desired.size() << " sizes not compatible";
        exit(1);
    }
}

/*
 * Returns the delta error from the layer z
 */
double_v Network::quadratic_delta(const double_v& zs, const double_v& actual, const double_v& desired) {
    if (zs.size() == actual.size() == desired.size()) {
        double_v result;
        for (double_v::const_iterator z = zs.begin(), a = actual.begin(), d = desired.begin();
                z != zs.end() && a != actual.end() && d != desired.end(); ++z, ++a, ++d) {
            result.push_back(
                (*a - *d) * Network::sigmoid_prime(*z));
        }
        return result;
    } else {
        logger(logERROR) << "Error in computing quadratic delta: "
            << zs.size() << ", " << actual.size() << ", and " << desired.size() << " sizes not compatible";
        exit(1);
    }
}

/*
 * Cross entropy cost function
 * Returns the cost with actual output and desired output
 */
double Network::cross_entropy_cost(const double_v& actual, const double_v& desired) {
    if (actual.size() == desired.size()) {
        double total = 0;
        for (double_v::const_iterator a = actual.begin(), d = desired.begin();
                a != actual.end() && d != desired.end(); ++a, ++d) {
            double val;
            try {
                val = (- *d) * std::log(*a) - (1 - *d) * std::log(1 - *a);
            } catch (std::domain_error e) {
                val = 0;
            }
            total += val;
        }
        return total;
    } else {
        logger(logERROR) << "Error in computing cross entropy cost: "
            << actual.size() << " and " << desired.size() << " sizes not compatible";
        exit(1);
    }
}

/*
 * Returns the delta error
 * First argument is a dummy argument to enforce consistency
 */
double_v Network::cross_entropy_delta(const double_v&, const double_v& actual, const double_v& desired) {
    if (actual.size() == desired.size()) {
        double_v result;
        for (double_v::const_iterator a = actual.begin(), d = desired.begin();
                a != actual.end() && d != desired.end(); ++a, ++d) {
            result.push_back(*a - *d);
        }
        return result;
    } else {
        logger(logERROR) << "Error in computing cross entropy delta: "
            << actual.size() << " and " << desired.size() << " sizes not compatible";
        exit(1);
    }
}

/*
 * Save the network to a file
 */
void Network::save(std::string filename) {
    std::ofstream out;
    out.open(filename);
    out << "biases\n";
    for (auto iter = this->biases.begin(); iter != this->biases.end(); ++iter) {
        out << (*iter)->to_string() << "\n";
    }
    out << "weights\n";
    for (auto iter = this->weights.begin(); iter != this->weights.end(); ++iter) {
        out << (*iter)->to_string() << "\n";
        
    }
    out << "end\n";
    
    out.close();
}
}  // namespace network

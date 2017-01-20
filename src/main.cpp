/* Copyright 2017 Alexander Burton. All rights reserved */

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "network.hpp"

#define tab '\t'
#define PARAM_SIZE 1

using std::cout;
using std::cerr;
using std::cin;
using std::endl;

void print_usage();
void print_short_usage();
int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv + 1, argv + argc);

    /* Help menu */
    if (std::find(args.begin(), args.end(), "-h") != args.end() ||
            std::find(args.begin(), args.end(), "--help") != args.end()) {
        print_usage();
        exit(0);
    }

    if (args.size() < PARAM_SIZE) {
        print_short_usage();
        exit(0);
    }
    if ((args.size() - PARAM_SIZE) % 2 == 1) {
        cerr << "Option missing parameter" << endl;
        exit(1);
    }

    bool has_test = false;
    std::string test_file;

    double learning_rate = 1.0;
    size_t hidden_layer_size = 10;
    size_t epoch_count = 1;
    size_t mini_batch_size = 1;

    bool has_load = false;
    std::string load;

    bool has_output = false;
    std::string output;

    for (size_t i = PARAM_SIZE; i < args.size() ; i += 2) {
        std::string arg = args[i];
        if (arg == "--testing-file" || arg == "-t") {
            has_test = true;
            test_file = args[i+1];
        } else if (arg == "--learning-rate" || arg == "-r") {
            learning_rate = std::stod(args[i+1]);
        } else if (arg == "--hidden-layer-size" || arg == "-s") {
            hidden_layer_size = std::stoul(args[i+1]);
        } else if (arg == "--epoch-count" || arg == "-e") {
            epoch_count = std::stoul(args[i+1]);
        } else if (arg == "--mini-batch-size" || arg == "-m") {
            mini_batch_size = std::stof(args[i+1]);
        } else if (arg == "--output" || arg == "-o") {
            has_output = true;
            output = args[i+1];
        } else if (arg == "--load" || arg == "-l") {
            has_load = true;
            load = args[i+1];
        } else {
            cerr << "Invalid option flag: " << arg << endl;
            exit(1);
        }
    }
    network::Network* net;

    if (has_load) {
        /* Load from file */
        net = new network::Network(load);
    } else {
        /* Initialize biases and weights using Gaussian distribution */
        std::vector<size_t> sizes;
        sizes.push_back(784);
        sizes.push_back(hidden_layer_size);
        sizes.push_back(10);
        net = new network::Network(sizes);
    }

    std::vector<mnist::dataset*> train_cases = mnist::load_test_cases(args[0]);
    std::vector<mnist::dataset*> test_cases;
    if (has_test) {
        test_cases = mnist::load_test_cases(args[0]);
    }

    net->SGD(&train_cases, &test_cases, epoch_count, mini_batch_size, learning_rate);
    delete net;
}

void print_short_usage() {
    cout << "Usage: network <training file> [options]" << endl;
}

void print_usage() {
    print_short_usage();
    cout <<
    "Parameters:" << endl <<
    tab << "training file           The CSV file to train the network" << endl <<
    "Options:" << endl <<
    tab << "--help -h               Print this help" << endl <<
    tab << "--learning-rate -r      The learning rate, eta" << endl <<
    tab << "--hidden-layer-size -s  The size of the hidden layer, defaults to 10" << endl <<
    tab << "--testing-file -t       Use this file for testing and tracking progress" << endl <<
    tab << "--epoch-count -e        Number of epochs, defaults to 1" << endl <<
    tab << "--mini-batch-size -m    Size of mini batch, defaults to 1" << endl <<
    tab << "--output -o             Saves the resulting neural network to a file" << endl <<
    tab << "--load -l               Loads the network from a previously saved file" << endl;
}

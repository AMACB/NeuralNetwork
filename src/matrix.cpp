/* Copyright 2017 Alexander Burton. All rights reserved. */

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "matrix.h"

namespace network {
/* Constructs a blank matrix */
Matrix::Matrix() {
    this->rows = 0;
    this->cols = 0;
    this->num_elements = 0;
    data.resize(0);
}

/* Constructs an m x n matrix */
Matrix::Matrix(const size_t& m, const size_t& n) {
    this->rows = m;
    this->cols = n;
    this->num_elements = m*n;
    data.resize(num_elements);
}

/* Constructs a row matrix from a vector */
Matrix::Matrix(const double_v& vec) {
    this->cols = vec.size();
    this->rows = 1;
    this->num_elements = vec.size();
    this->data = std::vector<double>(vec.begin(), vec.end());
}

/* Copy constructor */
Matrix::Matrix(const Matrix& other) {
    this->rows = other.rows;
    this->cols = other.cols;
    this->num_elements = other.num_elements;
    this->data = std::vector<double>(other.data.begin(), other.data.end());
}

/* From string representation */
Matrix::Matrix(const std::string& str) {
    size_t num_open_bracket = 0, num_close_bracket = 0, num_comma = 0;
    if (str.size() < 2) {
        throw std::runtime_error("Error in parsing matrix: string too short");
    }
    for (auto iter = str.begin(); iter != str.end(); ++iter) {
        if (*iter == ',') ++num_comma;
        if (*iter == '[') ++num_open_bracket;
        if (*iter == ']') ++num_close_bracket;
    }
    if (num_open_bracket != num_close_bracket) {
        throw std::runtime_error("Error in parsing matrix: missing [ or ]");
    }
    size_t num_rows = 0, num_cols = 0;
    if (num_open_bracket < 1) {
        throw std::runtime_error("Error in parsing matrix: could not find beginning [");
    }
    num_rows = num_open_bracket - 1;
    if (num_comma + 1 < num_rows) {
        throw std::runtime_error("Error in parsing matrix: missing ,");
    }
    // Num_cols will be of the form: row * (x-1) = k = (num_comma - num_rows + 1)
    // First find k
    size_t k = (num_comma + 1) - num_rows;
    // Make sure k % rows
    if (num_rows == 0) {
        num_cols = 0;
    } else if (k % num_rows != 0) {
        throw std::runtime_error("Error in parsing matrix: jagged or missing ,");
    }
    // Store solution
    num_cols = k / num_rows + 1;

    this->rows = num_rows;
    this->cols = num_cols;
    this->num_elements = num_rows * num_cols;
    // std::cout << this->rows << " " << this->cols << " " << this->num_elements << std::endl << std::flush;
    data.resize(this->num_elements);

    std::istringstream ss(str.substr(1, str.size() - 2));    
    std::string element;
    ssize_t row_number = -1;  // <-- 0 refers to the first, 1 to the second, etc.
    ssize_t col_number = 0;  // <-'
    while (std::getline(ss, element, ',')) {
        //continue;
        if (element.size() == 0) {
            throw std::runtime_error("Error in parsing matrix: expected value");
        }
        if (element.at(0) == '[') {
            col_number = 0;
            ++ row_number;  // increments for the first time, so -1 becomes 0 the first iteration
            element.erase(0, 1);
        } else if (element.back() == ']') {
            element.erase(element.size() - 1, 1);
        }
        this->set_index(static_cast<size_t>(row_number), static_cast<size_t>(col_number), std::stod(element));

        ++ col_number;
    }
}

/* Frees up all memory */
Matrix::~Matrix() {}

/* Prints the data stored in the matrix */
void Matrix::print() const {
    std::cout << "[";
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "[";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << this->index(i, j) << ",";
        }
        std::cout << "]," << std::endl;
    }
    std::cout << "]" << std::endl;
}

/* Returns a numpy.array string of the Matrix */
std::string Matrix::to_string() const {
    std::string s;
    s += "numpy.array([";
    for (size_t i = 0; i < this->rows; ++i) {
        s += "[";
        for (size_t j = 0; j < this->cols; ++j) {
            s += std::to_string(this->index(i, j));
            if (j < this->cols - 1) s += ",";
        }
        s += "]";
        if (i < this->rows - 1) s += ",";
    }
    s += "])";
    return s;
}

/* Returns a string representation of the dimensions of the matrix */
std::string Matrix::sizes() const {
    return std::to_string(this->rows) + "," + std::to_string(this->cols);
}

/* Sets all values to 0 */
void Matrix::zeroify() {
    std::fill(this->data.begin(), this->data.end(), 0);
}

/* Returns a transposed version of the Matrix */
Matrix Matrix::transposed() const {
    Matrix result(this->cols, this->rows);
    size_t k = 0;
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i, ++k) {
            result.data[k] = this->index(i, j);
        }
    }
    return result;
}

/* Transposes the matrix */
void Matrix::transpose() {
    double_v new_data;
    for (size_t j = 0; j < cols; ++j) {
        for (size_t i = 0; i < rows; ++i) {
            new_data.push_back(this->index(i, j));
        }
    }
    size_t tmp_cols = cols;
    this->cols = rows;
    this->rows = tmp_cols;
    this->data = new_data;
}

/* Flattens the matrix into a vector */
double_v Matrix::flatten() const {
    return this->data;
}

/* Scalar multiplication */
Matrix Matrix::operator*(double val) const {
    Matrix result(this->rows, this->cols);
    for (size_t i = 0; i < this->data.size(); ++i) {
        result.data[i] = this->data[i] * val;
    }
    return result;
}

/* Compound scalar multiplication */
void Matrix::operator*=(double val) {
    for (size_t i = 0; i < this->data.size(); ++i) {
        this->data[i] *= val;
    }
}

/* Inner product of matrices */
Matrix Matrix::inner_product(const Matrix& a, const Matrix& b) {
    if (a.rows == b.rows && a.cols == b.cols) {
        /* Element-wise product */
        Matrix result(a.rows, a.cols);
        for (size_t i = 0; i < a.num_elements && i < b.num_elements; ++i) {
            result.data.at(i) = a.data.at(i) * b.data.at(i);
        }
        return result;
    } else if (b.rows == 1 && a.cols == b.cols) {
        /* Treat the second as a vector; then use row-wise multiplication */
        Matrix result(a.cols, b.cols);
        for (size_t i = 0; i < a.rows; ++i) {
            for (size_t j = 0; j < a.cols && j < b.cols; ++j) {
                result.set_index(i, j, a.index(i, j)*b.index(0, j));
            }
        }
        return result;
    } else {
        std::stringstream err;
        err << "Cannot take inner product of matrices: sizes "
            << a.sizes() << " and " << b.sizes() << " not compatible";
        throw std::runtime_error(err.str());
    }
}

/* Vector times matrix */
Matrix Matrix::operator*(const double_v& vec) const {
    Matrix result(this->rows, this->cols);
    if (this->cols == vec.size()) {
        /* Multiply element-wise through rows */
        for (size_t j = 0; j < this->cols; ++j) {
            for (size_t i = 0; i < this->rows; ++i) {
                result.set_index(i, j, this->index(i, j) * vec.at(j));
            }
        }
    } else if (this->rows == vec.size()) {
        /* Multiply element-wise through columns */
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                result.set_index(i, j, this->index(i, j) * vec.at(i));
            }
        }
    } else {
        std::stringstream err;
        err << "Cannot take product of matrix to vector: sizes "
            << this->sizes() << " and " << vec.size() << " not compatible";
        throw std::runtime_error(err.str());
    }
    return result;
}

/* Multiplies two matrices */
Matrix Matrix::operator*(const Matrix& other) const {
    if (this->cols == other.rows) {
        /* Standard matrix multiplication: (m x n) times (n x p) = (m x p) */
        Matrix result(this->rows, other.cols);
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                double sum = 0;
                for (size_t k = 0; k < this->cols && k < other.rows; ++k) {
                    sum += (this->index(i, k)*other.index(k, j));
                }
                result.set_index(i, j, sum);
            }
        }
        return result;
    } else {
        std::stringstream err;
        err << "Cannot multiply matrices: sizes " << this->sizes()
            << " and " << other.sizes() << " not compatible";
        throw err.str();
    }
}

/* Compound addition-assignment */
void Matrix::operator+=(const Matrix& other) {
    if (this->cols == other.cols && this->rows == other.rows) {
        /* Element-wise addition */
        for (size_t i = 0; i < this->num_elements; ++i) {
            this->data[i] += other.data[i];
        }
    } else {
        std::stringstream err;
        err << "Cannot compound add matrices: sizes " << this->sizes()
            << " and " << other.sizes() << " not compatible";
        throw std::runtime_error(err.str());
    }
}

/* Adds matrices by element */
Matrix Matrix::operator+(const Matrix& other) const {
    if (this->cols == other.cols && this->rows == other.rows) {
    /* Element-wise addition */
        Matrix result(this->rows, this->cols);
        for (size_t i = 0; i < this->num_elements; ++i) {
            result.data[i] = this->data[i] + other.data[i];
        }
        return result;
    } else {
        std::stringstream err;
        err << "Cannot add matrices: sizes " << this->sizes()
            << " and " << other.sizes() << " not compatible";
        throw std::runtime_error(err.str());
    }
}

/* Adds the vectors by element */
/* Converts this Matrix to a vector if possible; else throws error*/
Matrix Matrix::operator+(const double_v& other) const {
    if ((this->cols == 1 && this->rows == other.size()) || (this->rows == 1 && this->cols == other.size())) {
        /* Runs as if this is a vector */
        Matrix result(this->rows, this->cols);
        for (size_t i = 0; i < this->data.size() && i < other.size(); ++i) {
            result.data[i] = this->data[i] + other[i];
        }
        return result;
    } else {
        std::stringstream err;
        err << "Cannot add matrix to vector: sizes " << this->sizes()
            << " and " << other.size() << " not compatible";
        throw std::runtime_error(err.str());
    }
}

/* Gets the value at the location */
double Matrix::index(size_t row, size_t col) const {
    return data[row*cols + col];
}

/* Sets the value at the location */
void Matrix::set_index(size_t row, size_t col, double val) {
    data[row*cols + col] = val;
}

/*
 * Convert to string 
 * String will be of the form [[a,b,c,...],[d,e,f,...],...]
 */
std::string Matrix::to_string() {
    std::stringstream str;
    str << "[";
    for (size_t i = 0; i < this->rows; ++i) {
        str << "[";
        for (size_t j = 0; j < this->cols; ++j) {
            str << this->index(i,j);
            if (j < this->cols - 1) str << ",";
        }
        str << "]";
        if (i < this->rows - 1) str << ",";
    }
    str << "]";
    return str.str();
}
}  // namespace network

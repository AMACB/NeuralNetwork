/* Copyright 2017 Alexander Burton. All rights reserved */

#ifndef SRC_MATRIX_H_
#define SRC_MATRIX_H_

#include <string>
#include <vector>

#include "log/logging.h"

typedef double ddouble;
typedef std::vector<ddouble> double_v;

namespace network {
class Matrix {
 public:
    size_t rows, cols;
    size_t num_elements;
    double_v data;

    /* Constructs a blank matrix */
    Matrix();

    /* Constructs an m x n matrix */
    Matrix(const size_t&, const size_t&);

    /* Constructs a row matrix from a vector */
    Matrix(const double_v&);

    /* Copy constructor */
    Matrix(const Matrix&);

    /* From string representation */
    Matrix(const std::string&);

    /* Frees up all memory */
    ~Matrix();

    /* Prints the data stored in the matrix */
    void print() const;

    /* Returns a numpy.array string of the Matrix */
    std::string to_string() const;

    /* Returns a string representation of the dimensions of the matrix */
    std::string sizes() const;

    /* Sets all values to 0 */
    void zeroify();

    /* Returns a transposed version of the Matrix */
    Matrix transposed() const;

    /* Transposes the matrix */
    void transpose();

    /* Flattens the matrix into a vector */
    double_v flatten() const;

    /* Scalar multiplication */
    Matrix operator*(double) const;

    /* Compound scalar multiplication */
    void operator*=(double);

    /* Inner product of matrices */
    static Matrix inner_product(const Matrix&, const Matrix&);

    /* Vector times matrix */
    Matrix operator*(const double_v&) const;

    /* Multiplies two matrices */
    Matrix operator*(const Matrix&) const;

    /* Compound addition-assignment */
    void operator+=(const Matrix&);

    /* Adds matrices by element */
    Matrix operator+(const Matrix&) const;

    /* Adds the vectors by element */
    /* Converts this Matrix to a vector if possible; else throws error*/
    Matrix operator+(const double_v&) const;

    /* Gets the value at the location */
    double index(size_t, size_t) const;

    /* Sets the value at the location */
    void set_index(size_t, size_t, double);
    
    /*
     * Convert to string 
     * String will be of the form [[a,b,c,...],[d,e,f,...],...]
     */
    std::string to_string();
};
}  // namespace network


#endif  // SRC_MATRIX_H_

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

typedef double ddouble;
typedef std::vector<ddouble> double_v;

/* Matrix class. Stores data like a matrix */
class Matrix {
public:
	size_t rows, cols;
	size_t num_elements;
	double_v data;
	
	/* Constructs a blank matrix */
	Matrix() {
		Matrix::Matrix(0,0);
	}

	/* Constructs an m x n matrix */
	Matrix(size_t m, size_t n) {
		this->rows = m;
		this->cols = n;
		this->num_elements = m*n;
		data.resize(num_elements);
	}

	/* Constructs a row matrix from a vector */
	Matrix(double_v vec) {
		this->cols = vec.size();
		this->rows = 1;
		this->num_elements = vec.size();
		this->data = vec;
	}

	/* Frees up all memory */
	~Matrix() {
	}

	/* Prints the data stored in the matrix */	
	void print() const {
		std::cout << "[";
		for (int i = 0; i < rows; ++i) {
			std::cout << "[";
			for (int j = 0; j < cols; ++j) {
				std::cout << this->index(i,j) << ",";
			}
			std::cout << "]," << std::endl;
		}
		std::cout << "]" << std::endl;
	}

	/* Returns a numpy.array string of the Matrix */
	std::string to_string() const {
		std::string s;
		s += "numpy.array([";
		for (int i = 0; i < this->rows; ++i) {
			s += "[";
			for (int j = 0; j < this->cols; ++j) {
				s += std::to_string(this->index(i,j));
				if (j < this->cols - 1) s += ",";
			}
			s += "]";
			if (i < this->rows - 1) s += ",";
		}
		s += "])";
		return s;
	}

	/* Returns a string representation of the dimensions of the matrix */
	std::string sizes() const {
		return std::to_string(this->rows) + "," + std::to_string(this->cols);
	}

	/* Sets all values to 0 */
	void zeroify() {
		std::fill(this->data.begin(), this->data.end(), 0);
	}

	/* Returns a transposed version of the Matrix */
	Matrix transposed() const {
		Matrix result(this->cols, this->rows);
		double_v new_data;
		for (int j = 0; j < cols; ++j) {
			for (int i = 0; i < rows; ++i) {
				new_data.push_back(this->index(i,j));
			}
		}
		result.data = new_data;
		return result;
	}

	/* Transposes the matrix */
	void transpose() {
		double_v new_data;
		for (int j = 0; j < cols; ++j) {
			for (int i = 0; i < rows; ++i) {
				new_data.push_back(this->index(i,j));
			}
		}
		int tmp_cols = cols;
		this->cols = rows;
		this->rows = tmp_cols;
		this->data = new_data;
	}

	/* Flattens the matrix into a vector */
	double_v flatten() const {
		return this->data;
	}

	/* Scalar multiplication */
	Matrix operator*(double val) const {
		Matrix result(this->rows, this->cols);
		for (int i = 0; i < this->data.size(); ++i) {
			result.data[i] = this->data[i] * val;
		}
		return result;
	}

	/* Inner product of matrices */
	static Matrix inner_product(Matrix a, Matrix b) {
		if (a.rows == b.rows && a.cols == b.cols) {
			Matrix result(a.rows, a.cols);
			for (int i = 0; i < a.num_elements && i < b.num_elements; ++i) {
				result.data.at(i) = a.data.at(i) * b.data.at(i);
			}
			return result;
		} else if (b.rows == 1 && a.cols == b.cols) {
			Matrix result(a.cols, b.cols);
			for (int i = 0; i < a.rows; ++i) {
				for (int j = 0; j < a.cols && j < b.cols; ++j) {
					result.set_index(i,j,a.index(i,j)*b.index(0,j));
				}
			}
			return result;
		} else {
			std::cerr << "Cannot take inner product of matrices: sizes " << a.sizes() << " and " << b.sizes() << " not compatible" << std::endl;
			throw 1;
		}
	}

	/* Vector times matrix */
	Matrix operator*(double_v vec) const {
		Matrix result(this->rows, this->cols);
		if (this->cols == vec.size()) {
			for (int j = 0; j < this->cols; ++j) {
				for (int i = 0; i < this->rows; ++i) {
					result.set_index(i,j,this->index(i,j) * vec.at(j));
				}
			}
		} else if (this->rows == vec.size()) {
			for (int i = 0; i < this->rows; ++i) {
				for (int j = 0; j < this->cols; ++j) {
					result.set_index(i,j,this->index(i,j) * vec.at(i));
				}
			}
		} else {
			std::cerr << "Cannot take product of matrix to vector: sizes " << this->sizes() << " and " << vec.size() << " not compatible" << std::endl;
			throw 1;
		}
		return result;
	}

	/* Multiplies two matrices */
	Matrix operator*(const Matrix& other) const {
		if (this->cols == other.rows) {		
			Matrix result(this->rows, other.cols);
			for (int i = 0; i < this->rows; ++i) {
				for (int j = 0; j < other.cols; ++j) {
					double sum = 0;
					for (int k = 0; k < this->cols && k < other.rows; ++k) {
						sum += (this->index(i,k)*other.index(k,j));
					}
					result.set_index(i,j,sum);
				}
			}
			return result;
		} /*else if (this->rows == other.cols) {
			return (this->transposed()) * (other.transposed());
		}*/ else {
			std::cerr << "Cannot multiply matrices: sizes " << this->sizes() << " and " << other.sizes() << " not compatible" << std::endl;
			throw 1;
		}
	}

	/* Compound addition-assignment */
	void operator+=(Matrix other) {
		if (this->cols == other.cols && this->rows == other.rows) {
			for (int i = 0; i < this->num_elements; ++i) {
				this->data[i] += other.data[i];
			}
		} else {
			std::cerr << "Cannot compound add matrices: sizes " << this->sizes() << " and " << other.sizes() << " not compatible" << std::endl;
			throw 1;
		}
	}

	/* Adds matrices by element */
	Matrix operator+(Matrix other) {
		if (this->cols == other.cols && this->rows == other.rows) {
			Matrix result(this->rows, this->cols);
			for (int i = 0; i < this->num_elements; ++i) {
				result.data[i] = this->data[i] + other.data[i];
			}
			return result;
		} else {
			std::cerr << "Cannot add matrices: sizes " << this->sizes() << " and " << other.sizes() << " not compatible" << std::endl;
			throw 1;
		}
	}

	/* Adds the vectors by element */
	/* Converts this Matrix to a vector if possible; else throws error*/
	Matrix operator+(double_v other) {
		double_v vals;
		if ((this->cols == 1 && this->rows == other.size()) || (this->rows == 1 && this->cols == other.size())) {
			vals = this->flatten();
			double_v result;
			for (int i = 0; i < vals.size() && i < other.size(); ++i) {
				result.push_back(vals[i] + other[i]);
			}
			return Matrix(result);
		} else if (this->cols == other.size()) {
			/* Add the vector to each row component-wise */
			Matrix result(this->rows, this->cols);
			for (int i = 0; i < this->rows; ++i) {
				for (int j = 0; j < this->cols && j < other.size(); ++j) {
					result.set_index(i,j,this->index(i,j) + other[j]);
				}
			}
			return result;
		} else if (this->rows == other.size()) {
			/* Add the vector to each column component-wise */
			Matrix result(this->rows, this->cols);
			for (int i = 0; i < this->cols; ++i) {
				for (int j = 0; j < this->rows && j < other.size(); ++j) {
					result.set_index(i,j,this->index(i,j) + other[j]);
				}
			}
			return result;
		} else {
			std::cout << "Cannot add matrix to vector: sizes " << this->sizes() << " and " << other.size() << " not compatible" << std::endl;
			throw 1;
		}	
	}

	/* Gets the value at the location */
	double index(int row, int col) const {
		return data[row*cols + col];
	}

	/* Sets the value at the location */
	void set_index(int row, int col, double val) {
		data[row*cols + col] = val;
	}
};

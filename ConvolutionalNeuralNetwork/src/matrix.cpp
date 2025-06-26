#include "../include/matrix.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <omp.h>

using namespace std;

long double random(double min, double max) {
    return min + (max - min) * ((long double)rand() / RAND_MAX);
}

long double sigmoid(long double x, bool derivative) {
    if (derivative) {
        long double s = sigmoid(x);
        return s * (1 - s);
    }
    return 1.0L / (1.0L + exp(-x));
}

long double relu(long double x, bool derivative) {
    return derivative ? (x > 0 ? 1.0L : 0.0L) : max(0.0L, x);
}

Matrix::Matrix() : cols(0), rows(0) {}

Matrix::Matrix(size_t m_rows, size_t m_cols, long double el)
: matrix(m_rows * m_cols, el), cols(m_cols), rows(m_rows) {}


std::ostream& operator<<(std::ostream& out, const Matrix& matr) {
    for (size_t i = 0; i < matr.rows; ++i) {
        for (size_t j = 0; j < matr.cols; ++j)
            out << matr(i, j) << ' ';
        out << '\n';
    }
    return out;
}

long double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols)
        throw out_of_range("Incorect index");
    return matrix[row * cols + col];
}
    
const long double Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols)
        throw out_of_range("Incorect index");
    return matrix[row * cols + col];
}

size_t Matrix::rows_() const { return rows; }
size_t Matrix::cols_() const { return cols; }

Matrix Matrix::T() const {
    Matrix result(cols, rows);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result(j, i) = (*this)(i, j);
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (cols != other.cols || rows != other.rows)
        throw out_of_range("Different dimensions during +");

    Matrix result(rows, cols);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result(i, j) = (*this)(i, j) + other(i, j);
    return result;
}

Matrix& Matrix::operator+=(const Matrix& other) {
    if (cols != other.cols || rows != other.rows)
        throw out_of_range("Different dimensions during +=");

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            (*this)(i, j) += other(i, j);
    return *this;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (cols != other.cols || rows != other.rows)
        throw out_of_range("Different dimensions during -");

    Matrix result(rows, cols);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result(i, j) = (*this)(i, j) - other(i, j);
    return result;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (cols != other.cols || rows != other.rows)
        throw out_of_range("Different dimensions during -=");

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            (*this)(i, j) -= other(i, j);
    return *this;
}

Matrix Matrix::operator^(const Matrix& other) const {
    if (cols != other.cols || rows != other.rows)
        throw out_of_range("Different dimensions during ^");

    Matrix result(rows, cols);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result(i, j) = (*this)(i, j) * other(i, j);
    return result;
}

Matrix Matrix::operator*(const long double scalar) const {
    Matrix result(rows, cols);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result(i, j) = (*this)(i, j) * scalar;
    return result;
}

Matrix Matrix::operator/=(const long double scalar) const {
    if (scalar == 0)
        throw invalid_argument("Division by zero");

    Matrix result(rows, cols);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result(i, j) = (*this)(i, j) / scalar;
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows)
        throw invalid_argument("Invalid dimensions for multiplication");

    Matrix result(rows, other.cols);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < other.cols; j++) {
            long double sum = 0;
            for (size_t k = 0; k < cols; k++)
                sum += (*this)(i, k) * other(k, j);
            result(i, j) = sum;
        }
    }
    return result;
}

void Matrix::save_to_bin_file(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    
    out.write(reinterpret_cast<const char*>(matrix.data()), 
            matrix.size() * sizeof(long double));
}

void Matrix::randomize(double min, double max) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<long double> distrib(min, max);

        #pragma omp for collapse(2)
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                (*this)(i, j) = distrib(gen);
            }
        }
    }
}

Matrix Matrix::sigmoid(bool is_derivative) const {
    Matrix result(rows, cols);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result(i, j) = ::sigmoid((*this)(i, j), is_derivative);
    return result;
}

Matrix Matrix::ReLU(bool is_derivative) const {
    Matrix result(rows, cols);
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            result(i, j) = ::relu((*this)(i, j), is_derivative);
    return result;
}

long double Matrix::cross_entropy(const Matrix& ideal) const {
    if (rows != ideal.rows || cols != ideal.cols)
        throw invalid_argument("Matrix sizes must match");

    long double res = 0.0L;
    constexpr long double EPS = 1e-15L;

    #pragma omp parallel for collapse(2) reduction(+:res)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            long double prob = std::max((*this)(i, j), EPS);
            res += -1.0L * ideal(i, j) * log(prob);
        }
    }

    return res;
}

Matrix Matrix::softmax() const {
    Matrix result(rows, cols);

    #pragma omp parallel for
    for (size_t i = 0; i < rows; i++) {
        long double max_val = (*this)(i, 0);
        for (size_t j = 1; j < cols; j++) {
            if ((*this)(i, j) > max_val) {
                max_val = (*this)(i, j);
            }
        }

        long double sum_exp = 0.0L;
        for (size_t j = 0; j < cols; j++) {
            result(i, j) = exp((*this)(i, j) - max_val);
            sum_exp += result(i, j);
        }

        for (size_t j = 0; j < cols; j++) {
            result(i, j) /= sum_exp;
        }
    }

    return result;
}

vector<long double> Matrix::vectorise() const {
    return matrix;
}

Matrix Matrix::ravel() {
    Matrix ravel (1, cols * rows);

    #pragma omp parallel for
    for (int i = 0; i < cols * rows; i++)
        ravel(0, i) = matrix[i];

    return ravel;
}

Matrix Matrix::unravel(size_t new_cols) {
    if (cols % new_cols != 0) 
        throw invalid_argument("Invalid new cols");

    Matrix unravel(cols / new_cols, new_cols);

    for (int i = 0; i < cols / new_cols; i++) {
        for (int j = 0; j < new_cols; j++) {
            unravel(i, j) = matrix[i + j];
        }
    }

    return unravel;
}

Matrix Matrix::add_padding(size_t m, size_t n) {
    Matrix padded_matrix(rows + m, cols + n);
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            padded_matrix(i + (m / 2), j + (n / 2)) = (*this)(i, j);

    return padded_matrix;
}

Matrix Matrix::convolve(const Matrix& kernel, bool same_padding) {
    if (kernel.rows == 0 || kernel.cols == 0)
        throw std::invalid_argument("Kernel must be non-empty");

    size_t pad_rows = same_padding ? kernel.rows - 1 : 0;
    size_t pad_cols = same_padding ? kernel.cols - 1 : 0;

    size_t padded_rows = rows + pad_rows;
    size_t padded_cols = cols + pad_cols;
    Matrix padded(padded_rows, padded_cols);

    size_t row_offset = pad_rows / 2;
    size_t col_offset = pad_cols / 2;

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            padded(i + row_offset, j + col_offset) = (*this)(i, j);
        }
    }

    size_t out_rows = same_padding ? rows : rows - kernel.rows + 1;
    size_t out_cols = same_padding ? cols : cols - kernel.cols + 1;
    Matrix result(out_rows, out_cols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < out_rows; i++) {
        for (size_t j = 0; j < out_cols; j++) {
            long double sum = 0.0L;

            for (size_t k_i = 0; k_i < kernel.rows; k_i++) {
                for (size_t k_j = 0; k_j < kernel.cols; k_j++) {
                    size_t pi = i + k_i;
                    size_t pj = j + k_j;
                    sum += padded(pi, pj) * kernel(k_i, k_j);
                }
            }

            result(i, j) = sum;
        }
    }

    return result;
}

Matrix Matrix::rot180() {
    Matrix flipped(rows, cols);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            flipped(i, j) = (*this)(rows - 1 - i, cols - 1 - j);

    return flipped;
}

using IndexMatrix = std::vector<std::vector<std::pair<size_t, size_t>>>;

std::pair<Matrix, IndexMatrix> Matrix::max_pooling(size_t m, size_t n) {
    if (rows % m != 0 || cols % n != 0) 
        throw invalid_argument("Incorrect max pooling size");

    Matrix pooled(rows / m, cols / n);
    IndexMatrix indices(rows / m, std::vector<std::pair<size_t, size_t>>(cols / n));

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; i += m) {
        for (size_t j = 0; j < cols; j += n) {
            long double max_val = this->operator()(i, j);
            size_t max_i = i;
            size_t max_j = j;

            for (size_t di = 0; di < m; di++) {
                for (size_t dj = 0; dj < n; dj++) {
                    size_t cur_i = i + di;
                    size_t cur_j = j + dj;
                    long double val = this->operator()(cur_i, cur_j);
                    if (val > max_val) {
                        max_val = val;
                        max_i = cur_i;
                        max_j = cur_j;
                    }
                }
            }
            size_t out_i = i / m;
            size_t out_j = j / n;
            pooled(out_i, out_j) = max_val;
            indices[out_i][out_j] = std::make_pair(max_i, max_j);
        }
    }

    return {pooled, indices};
}


Matrix Matrix::max_unpooling(const IndexMatrix& indices, size_t original_rows, size_t original_cols) {
    Matrix unpooled(original_rows, original_cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            auto [row, col] = indices[i][j];
            unpooled(row, col) = (*this)(i, j);
        }
    }

    return unpooled;
}

void Matrix::serialize(std::ostream& out) const {
    out << rows << " " << cols << "\n";
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            out << matrix[i * cols + j] << " ";
    out << "\n";
}

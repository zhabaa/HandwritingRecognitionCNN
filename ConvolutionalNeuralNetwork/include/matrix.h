// Matrix.h
#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <ostream>

long double random(double min = -1.0, double max = 1.0);
long double sigmoid(long double x, bool derivative = false);
long double relu(long double x, bool derivative = false);

class Matrix {
private:
    std::vector<long double> matrix;
    size_t cols;
    size_t rows;

public:
    Matrix();
    Matrix(size_t m_rows, size_t m_cols, long double el = 0.0L);

    friend std::ostream& operator<<(std::ostream& out, const Matrix& matr);
    long double& operator()(size_t row, size_t col);
    const long double operator()(size_t row, size_t col) const;

    size_t rows_() const;
    size_t cols_() const;

    Matrix T() const;

    Matrix operator+(const Matrix& other) const;
    Matrix& operator+=(const Matrix& other);
    Matrix operator-(const Matrix& other) const;
    Matrix& operator-=(const Matrix& other);
    Matrix operator^(const Matrix& other) const;
    Matrix operator*(const long double scalar) const;
    Matrix operator/=(const long double scalar) const;
    Matrix operator*(const Matrix& other) const;

    void save_to_bin_file(const std::string& filename) const;
    void randomize(double min = -1.0, double max = 1.0);

    Matrix sigmoid(bool is_derivative = false) const;
    Matrix ReLU(bool is_derivative = false) const;

    long double cross_entropy(const Matrix& ideal) const;
    Matrix softmax() const;

    std::vector<long double> vectorise() const;
    Matrix ravel();
    Matrix unravel(size_t new_cols);

    Matrix add_padding(size_t m, size_t n);
    Matrix convolve(const Matrix& kernel, bool same_padding = true);
    Matrix rot180();

    using IndexMatrix = std::vector<std::vector<std::pair<size_t, size_t>>>;
    std::pair<Matrix, IndexMatrix> max_pooling(size_t m, size_t n);
    Matrix max_unpooling(const IndexMatrix& indices, size_t original_rows, size_t original_cols);

    void serialize(std::ostream& out) const;
};

#endif // MATRIX_H

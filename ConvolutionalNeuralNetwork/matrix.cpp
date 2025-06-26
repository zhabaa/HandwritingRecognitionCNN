#include <iostream>
#include <vector>
#include <type_traits>
#include <random>
#include <iomanip>
#include <chrono>
#include <functional>
#include <omp.h>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <string>
#include <sstream>
#include <cassert>


#define CLASSES 10

using namespace std;

long double random(double min = -1.0, double max = 1.0) {
    return min + (max - min) * ((long double)rand() / RAND_MAX);
}

long double sigmoid(long double x, bool derivative = false) {
    if (derivative) {
        long double s = sigmoid(x);
        return s * (1 - s);
    }
    return 1.0L / (1.0L + exp(-x));
}

long double relu(long double x, bool derivative = false) {
    return derivative ? (x > 0 ? 1.0L : 0.0L) : max(0.0L, x);
}

class Matrix {
private:
    vector<long double> matrix;
    size_t cols;
    size_t rows;

public:
    Matrix() : rows(0), cols(0), matrix() {}

    Matrix(size_t m_rows, size_t m_cols,  long double el = 0.0L)
        : cols(m_cols), rows(m_rows), matrix(m_cols * m_rows, el) {}

    friend ostream& operator<<(ostream& out, const Matrix& matr) {
        for (size_t i = 0; i < matr.rows; i++) {
            for (size_t j = 0; j < matr.cols; j++) {
                out << matr(i, j) << " ";
            }
            out << endl;
        }
        return out;
    }

    long double& operator()(size_t row, size_t col) {
        if (row >= rows || col >= cols)
            throw out_of_range("Incorect index");
        return matrix[row * cols + col];
    }
    
    const long double operator()(size_t row, size_t col) const {
        if (row >= rows || col >= cols)
            throw out_of_range("Incorect index");
        return matrix[row * cols + col];
    }

    size_t rows_() const { return rows; }
    size_t cols_() const { return cols; }

    Matrix T() const {
        Matrix result(cols, rows);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(j, i) = (*this)(i, j);
        return result;
    }

    Matrix operator+(const Matrix& other) const {
        if (cols != other.cols || rows != other.rows)
            throw out_of_range("Different dimensions during +");

        Matrix result(rows, cols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = (*this)(i, j) + other(i, j);
        return result;
    }

    Matrix& operator+=(const Matrix& other) {
        if (cols != other.cols || rows != other.rows)
            throw out_of_range("Different dimensions during +=");

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                (*this)(i, j) += other(i, j);
        return *this;
    }

    Matrix operator-(const Matrix& other) const {
        if (cols != other.cols || rows != other.rows)
            throw out_of_range("Different dimensions during -");

        Matrix result(rows, cols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = (*this)(i, j) - other(i, j);
        return result;
    }

    Matrix& operator-=(const Matrix& other) {
        if (cols != other.cols || rows != other.rows)
            throw out_of_range("Different dimensions during -=");

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                (*this)(i, j) -= other(i, j);
        return *this;
    }

    Matrix operator^(const Matrix& other) const {
        if (cols != other.cols || rows != other.rows)
            throw out_of_range("Different dimensions during ^");

        Matrix result(rows, cols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = (*this)(i, j) * other(i, j);
        return result;
    }

    Matrix operator*(const long double scalar) const {
        Matrix result(rows, cols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = (*this)(i, j) * scalar;
        return result;
    }

    Matrix operator/=(const long double scalar) const {
        if (scalar == 0)
            throw invalid_argument("Division by zero");

        Matrix result(rows, cols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = (*this)(i, j) / scalar;
        return result;
    }

    Matrix operator*(const Matrix& other) const {
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

    void save_to_bin_file(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
        
        // Write matrix dimension
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        
        // Write matrix data
        out.write(reinterpret_cast<const char*>(matrix.data()), 
                matrix.size() * sizeof(long double));
    }

    void randomize(double min = -1.0, double max = 1.0) {
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

    Matrix sigmoid(bool is_derivative = false) const {
        Matrix result(rows, cols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = ::sigmoid((*this)(i, j), is_derivative);
        return result;
    }

    Matrix ReLU(bool is_derivative = false) const {
        Matrix result(rows, cols);
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                result(i, j) = ::relu((*this)(i, j), is_derivative);
        return result;
    }

    long double cross_entropy(const Matrix& ideal) const {
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

    Matrix softmax() const {
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

    vector<long double> vectorise() const {
        return matrix;
    }

    Matrix ravel() {
        Matrix ravel (1, cols * rows);

        #pragma omp parallel for
        for (int i = 0; i < cols * rows; i++)
            ravel(0, i) = matrix[i];

        return ravel;
    }

    Matrix unravel(size_t new_cols) {
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

    Matrix add_padding(size_t m, size_t n) {
        Matrix padded_matrix(rows + m, cols + n);
        
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                padded_matrix(i + (m / 2), j + (n / 2)) = (*this)(i, j);

        return padded_matrix;
    }

    Matrix convolve(const Matrix& kernel, bool same_padding = true) {
        if (kernel.rows == 0 || kernel.cols == 0)
            throw std::invalid_argument("Kernel must be non-empty");

        size_t pad_rows = same_padding ? kernel.rows - 1 : 0;
        size_t pad_cols = same_padding ? kernel.cols - 1 : 0;

        size_t padded_rows = rows + pad_rows;
        size_t padded_cols = cols + pad_cols;
        Matrix padded(padded_rows, padded_cols);

        // Copy original matrix to center of padded matrix
        size_t row_offset = pad_rows / 2;
        size_t col_offset = pad_cols / 2;

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                padded(i + row_offset, j + col_offset) = (*this)(i, j);
            }
        }

        // Determine output size
        size_t out_rows = same_padding ? rows : rows - kernel.rows + 1;
        size_t out_cols = same_padding ? cols : cols - kernel.cols + 1;
        Matrix result(out_rows, out_cols);

        // Perform convolution
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


    Matrix cross_correlate(const Matrix& kernel, size_t stride = 1, size_t padding = 0) {
        if (kernel.rows_() == 0 || kernel.cols_() == 0 || 
            kernel.rows_() > rows_() || kernel.cols_() > cols_()) {
            throw std::invalid_argument("Invalid kernel dimensions");
        }

        size_t out_rows = (rows_() + 2 * padding - kernel.rows_()) / stride + 1;
        size_t out_cols = (cols_() + 2 * padding - kernel.cols_()) / stride + 1;
        
        Matrix output(out_rows, out_cols);

        Matrix padded = *this;
        if (padding > 0) {
            padded = this->add_padding(padding, padding);
        }

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < out_rows; ++i) {
            for (size_t j = 0; j < out_cols; ++j) {
                long double sum = 0.0L;
                
                size_t x = i * stride;
                size_t y = j * stride;
                
                for (size_t ki = 0; ki < kernel.rows_(); ++ki) {
                    for (size_t kj = 0; kj < kernel.cols_(); ++kj) {
                        sum += padded(x + ki, y + kj) * kernel(ki, kj);
                    }
                }
                
                output(i, j) = sum;
            }
        }
        
        return output;
    }


    Matrix rot180() {
        Matrix flipped(rows, cols);

        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < rows; i++)
            for (size_t j = 0; j < cols; j++)
                flipped(i, j) = (*this)(rows - 1 - i, cols - 1 - j);

        return flipped;
    }

    using IndexMatrix = std::vector<std::vector<std::pair<size_t, size_t>>>;

   std::pair<Matrix, IndexMatrix> max_pooling(size_t m, size_t n) {
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


    Matrix max_unpooling(const IndexMatrix& indices, size_t original_rows, size_t original_cols) {
        Matrix unpooled(original_rows, original_cols);

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                auto [row, col] = indices[i][j];
                unpooled(row, col) = (*this)(i, j);
            }
        }

        return unpooled;
    }

    void serialize(std::ostream& out) const {
        out << rows << " " << cols << "\n";
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                out << matrix[i * cols + j] << " ";
        out << "\n";
    }
    
};

void readMNISTCSV(const std::string& filename, 
                 std::vector<Matrix>& images, 
                 std::vector<int>& labels) {
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    size_t line_count = 0;

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        std::getline(ss, value, ',');
        int label = std::stoi(value);
        labels.push_back(label);

        Matrix img(28, 28);

        for (size_t i = 0; i < 28; ++i) {
            for (size_t j = 0; j < 28; ++j) {
                if (!std::getline(ss, value, ',')) {
                    throw std::runtime_error("Invalid CSV format in file: " + filename);
                }
                img(i, j) = std::stod(value) / 255;
            }
        }

        images.push_back(img);
        line_count++;

        if (line_count % 1000 == 0) {
            std::cout << "Processed " << line_count << " images from " << filename << std::endl;
        }
    }

    std::cout << "Successfully loaded " << line_count << " images from " << filename << std::endl;
}

vector<Matrix> one_hot_encoding_labels(const vector<int>& labels) {
    vector<Matrix> one_hot_encoded_labels(labels.size());

    #pragma omp parallel for
    for (size_t i = 0; i < labels.size(); i++) {
        Matrix current_label(1, CLASSES, 0);
        current_label(0, labels[i]) = 1.L;
        one_hot_encoded_labels[i] = current_label;
    }

    return one_hot_encoded_labels;
}

void save_model(const Matrix& K1, const Matrix& K2,
                const Matrix& W1, const Matrix& W2,
                const Matrix& b1, const Matrix& b2,
                const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) throw std::runtime_error("Cannot open model file for writing");
    K1.serialize(out); K2.serialize(out);
    W1.serialize(out); W2.serialize(out);
    b1.serialize(out); b2.serialize(out);
    out.close();
}

#define PICTURE_DIM 28
#define KERNEL_DIM 3
#define POOLING_DIM 2

#define INPUT_DIM 49
#define OUT_DIM 10
#define H_DIM 128


#define LEARNING_RATE 0.0001
#define NUM_EPOCHS 400
#define BATCH_SIZE 50

int main() {
    try {
        std::vector<Matrix> train_images, test_images;
        std::vector<int> train_labels, test_labels;

        std::cout << "Loading training data..." << std::endl;
        readMNISTCSV("data/dataset/train/mnist_train.csv", train_images, train_labels);
        
        std::cout << "\nLoading test data..." << std::endl;
        readMNISTCSV("data/dataset/test/mnist_test.csv", test_images, test_labels);

        std::cout << "\nData loaded successfully:" << std::endl;
        std::cout << "Training set: " << train_images.size() << " images" << std::endl;
        std::cout << "Test set: " << test_images.size() << " images" << std::endl;

        cout << "Try to encode" << endl;

        vector<Matrix> train_labels_encoded = one_hot_encoding_labels(train_labels);
        vector<Matrix> test_labels_encoded = one_hot_encoding_labels(test_labels);

        cout << "Encoded" << endl;

        Matrix K_1(KERNEL_DIM, KERNEL_DIM);
        Matrix K_2(KERNEL_DIM, KERNEL_DIM);

        Matrix W_1(INPUT_DIM, H_DIM);
        Matrix W_2(H_DIM, OUT_DIM);
        
        Matrix b_1(1, H_DIM);
        Matrix b_2(1, OUT_DIM);

        K_1.randomize();
        K_2.randomize();

        W_1.randomize();
        W_2.randomize();

        b_1.randomize();
        b_2.randomize();

        Matrix U_1(PICTURE_DIM, PICTURE_DIM);
        Matrix V_1_(PICTURE_DIM, PICTURE_DIM);
        std::vector<std::vector<std::pair<size_t, size_t>>> IND_1;
        Matrix V_1(PICTURE_DIM / 2, PICTURE_DIM / 2);

        Matrix U_2(PICTURE_DIM / 2, PICTURE_DIM / 2);
        Matrix V_2_(PICTURE_DIM / 2, PICTURE_DIM / 2);
        std::vector<std::vector<std::pair<size_t, size_t>>> IND_2;
        Matrix V_2(PICTURE_DIM / 4, PICTURE_DIM / 4);

        Matrix T_1(1, H_DIM);
        Matrix H_1(1, H_DIM);

        Matrix T_2(1, OUT_DIM);
        Matrix Z(1, OUT_DIM);

        long double L;

        Matrix dL_dt2(1, OUT_DIM);
        Matrix dL_dw2(H_DIM, OUT_DIM);
        Matrix dL_db2(1, OUT_DIM);
        Matrix dL_dh1(1, H_DIM);
        Matrix dL_dt1(1, H_DIM);
        Matrix dL_dw1(INPUT_DIM, H_DIM);
        Matrix dL_db1(1, H_DIM);
        Matrix dL_dx_(1, INPUT_DIM);

        Matrix dL_dv2(PICTURE_DIM / 4, PICTURE_DIM / 4);
        Matrix dL_dv2_(PICTURE_DIM / 2, PICTURE_DIM / 2);
        Matrix dL_du2(PICTURE_DIM / 2, PICTURE_DIM / 2);
        Matrix dL_dv1(PICTURE_DIM / 2, PICTURE_DIM / 2);
        Matrix dL_dk2(KERNEL_DIM, KERNEL_DIM);
        Matrix dL_dv1_(PICTURE_DIM, PICTURE_DIM);
        Matrix dL_du1(PICTURE_DIM, PICTURE_DIM);
        Matrix dL_dk1(KERNEL_DIM, KERNEL_DIM);




        vector<long double> loss_arr;

        const int EARLY_STOPPING_PATIENCE = 3;
        const std::string MODEL_FILE = "model_weights.txt";
        const std::string BIN_MODEL_FILE = "model_weights.bin";
        long double best_loss = std::numeric_limits<long double>::max();
        int epochs_no_improve = 0;

        const int batch_size = 64;
        const int num_samples = 60000;

        for (int ep = 0; ep < NUM_EPOCHS; ep++) {
            long double epoch_loss = 0;

            for (int i = 0; i < num_samples; i += batch_size) {
                long double batch_loss = 0;

                int actual_batch_size = std::min(batch_size, num_samples - i);

                Matrix avg_dL_dw1(INPUT_DIM, H_DIM, 0.0);
                Matrix avg_dL_dw2(H_DIM, OUT_DIM, 0.0);
                Matrix avg_dL_db1(1, H_DIM, 0.0);
                Matrix avg_dL_db2(1, OUT_DIM, 0.0);
                Matrix avg_dL_dk1(KERNEL_DIM, KERNEL_DIM, 0.0);
                Matrix avg_dL_dk2(KERNEL_DIM, KERNEL_DIM, 0.0);

                for (int j = i; j < i + actual_batch_size; ++j) {
                    Matrix X = train_images[j];
                    Matrix Y = train_labels_encoded[j];

                    // --- Forward pass ---
                    U_1 = X.convolve(K_1);
                    V_1_ = U_1.ReLU();
                    std::tie(V_1, IND_1) = V_1_.max_pooling(POOLING_DIM, POOLING_DIM);

                    U_2 = V_1.convolve(K_2);
                    V_2_ = U_2.ReLU();
                    std::tie(V_2, IND_2) = V_2_.max_pooling(POOLING_DIM, POOLING_DIM);

                    Matrix X_ = V_2.ravel();
                    T_1 = X_ * W_1 + b_1;
                    H_1 = T_1.ReLU();
                    T_2 = H_1 * W_2 + b_2;
                    Z = T_2.softmax();

                    long double L = Z.cross_entropy(Y);
                    batch_loss += L;

                    // --- Backward pass ---
                    Matrix dL_dt2 = Z - Y;
                    Matrix dL_dw2 = H_1.T() * dL_dt2;
                    Matrix dL_db2 = dL_dt2;

                    Matrix dL_dh1 = dL_dt2 * W_2.T();
                    Matrix dL_dt1 = dL_dh1 ^ T_1.ReLU(true);
                    Matrix dL_dw1 = X_.T() * dL_dt1;
                    Matrix dL_db1 = dL_dt1;
                    Matrix dL_dx_ = dL_dt1 * W_1.T();

                    Matrix dL_dv2 = dL_dx_.unravel(PICTURE_DIM / 4);
                    Matrix dL_dv2_ = dL_dv2.max_unpooling(IND_2, PICTURE_DIM / 2, PICTURE_DIM / 2);
                    Matrix dL_du2 = dL_dv2_ ^ U_2.ReLU(true);
                    Matrix dL_dk2 = V_1.add_padding(KERNEL_DIM - 1, KERNEL_DIM - 1).convolve(dL_du2, false);

                    Matrix dL_dv1 = dL_du2.convolve(K_2.rot180());
                    Matrix dL_dv1_ = dL_dv1.max_unpooling(IND_1, PICTURE_DIM, PICTURE_DIM);
                    Matrix dL_du1 = dL_dv1_ ^ U_1.ReLU(true);
                    Matrix dL_dk1 = X.add_padding(KERNEL_DIM - 1, KERNEL_DIM - 1).convolve(dL_du1, false);

                    // --- Накопление градиентов ---
                    avg_dL_dw1 += dL_dw1;
                    avg_dL_db1 += dL_db1;
                    avg_dL_dw2 += dL_dw2;
                    avg_dL_db2 += dL_db2;
                    avg_dL_dk1 += dL_dk1;
                    avg_dL_dk2 += dL_dk2;
                }

                // --- Усреднение градиентов ---
                avg_dL_dw1 /= actual_batch_size;
                avg_dL_db1 /= actual_batch_size;
                avg_dL_dw2 /= actual_batch_size;
                avg_dL_db2 /= actual_batch_size;
                avg_dL_dk1 /= actual_batch_size;
                avg_dL_dk2 /= actual_batch_size;

                // --- Обновление весов ---
                W_1 -= avg_dL_dw1 * LEARNING_RATE;
                W_2 -= avg_dL_dw2 * LEARNING_RATE;
                b_1 -= avg_dL_db1 * LEARNING_RATE;
                b_2 -= avg_dL_db2 * LEARNING_RATE;
                K_1 -= avg_dL_dk1 * LEARNING_RATE;
                K_2 -= avg_dL_dk2 * LEARNING_RATE;

                epoch_loss += batch_loss;
            }

    epoch_loss /= (num_samples);

    std::cout << "Epoch " << ep << ", avg loss = " << epoch_loss << std::endl;

            // Early stopping logic
            if (epoch_loss < best_loss) {
                best_loss = epoch_loss;
                epochs_no_improve = 0;
                save_model(K_1, K_2, W_1, W_2, b_1, b_2, MODEL_FILE);
                K_1.save_to_bin_file(BIN_MODEL_FILE);
                K_2.save_to_bin_file(BIN_MODEL_FILE);
                W_1.save_to_bin_file(BIN_MODEL_FILE);
                W_2.save_to_bin_file(BIN_MODEL_FILE);
                b_1.save_to_bin_file(BIN_MODEL_FILE);
                b_2.save_to_bin_file(BIN_MODEL_FILE);
                std::cout << "Model improved. Saved to " << MODEL_FILE << std::endl;
            } else {
                epochs_no_improve++;
                if (epochs_no_improve >= EARLY_STOPPING_PATIENCE) {
                    std::cout << "Early stopping: no improvement for " << EARLY_STOPPING_PATIENCE << " epochs." << std::endl;
                    break;
                }
    }
        }

        // for (int i = 0; i < loss_arr.size(); i++) {
        //     cout << loss_arr[i] << endl;
        // }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
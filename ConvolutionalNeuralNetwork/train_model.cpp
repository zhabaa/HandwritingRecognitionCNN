#include "include/matrix.h"
#include "include/utilits.h"
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


using namespace std;

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
        // --- Load dataset ---
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

        // --- Use one-hot encoding to transform real answers into matrices ---
        vector<Matrix> train_labels_encoded = one_hot_encoding_labels(train_labels);
        vector<Matrix> test_labels_encoded = one_hot_encoding_labels(test_labels);

        cout << "Encoded" << endl;

        // --- Define kernels, weights and biases matrices ---
        Matrix K_1(KERNEL_DIM, KERNEL_DIM);
        Matrix K_2(KERNEL_DIM, KERNEL_DIM);

        Matrix W_1(INPUT_DIM, H_DIM);
        Matrix W_2(H_DIM, OUT_DIM);
        
        Matrix b_1(1, H_DIM);
        Matrix b_2(1, OUT_DIM);

        // --- Fill in all matrices with random values ---
        K_1.randomize();
        K_2.randomize();

        W_1.randomize();
        W_2.randomize();

        b_1.randomize();
        b_2.randomize();

        // --- Matrices to calculate the result for convolutional layers ---
        Matrix U_1(PICTURE_DIM, PICTURE_DIM); 
        Matrix V_1_(PICTURE_DIM, PICTURE_DIM);
        std::vector<std::vector<std::pair<size_t, size_t>>> IND_1;
        Matrix V_1(PICTURE_DIM / 2, PICTURE_DIM / 2);

        Matrix U_2(PICTURE_DIM / 2, PICTURE_DIM / 2);
        Matrix V_2_(PICTURE_DIM / 2, PICTURE_DIM / 2);
        std::vector<std::vector<std::pair<size_t, size_t>>> IND_2;
        Matrix V_2(PICTURE_DIM / 4, PICTURE_DIM / 4);

        // --- Matrices to calculate the result for fully-connected layers ---
        Matrix T_1(1, H_DIM);
        Matrix H_1(1, H_DIM);

        Matrix T_2(1, OUT_DIM);
        Matrix Z(1, OUT_DIM);

        // --- Varibale to calculate loss function ---
        long double L;

        // --- Convolutional layers gradients---
        Matrix dL_dt2(1, OUT_DIM);
        Matrix dL_dw2(H_DIM, OUT_DIM);
        Matrix dL_db2(1, OUT_DIM);
        Matrix dL_dh1(1, H_DIM);
        Matrix dL_dt1(1, H_DIM);
        Matrix dL_dw1(INPUT_DIM, H_DIM);
        Matrix dL_db1(1, H_DIM);
        Matrix dL_dx_(1, INPUT_DIM);

        // --- Fully-connected layers gradient---
        Matrix dL_dv2(PICTURE_DIM / 4, PICTURE_DIM / 4);
        Matrix dL_dv2_(PICTURE_DIM / 2, PICTURE_DIM / 2);
        Matrix dL_du2(PICTURE_DIM / 2, PICTURE_DIM / 2);
        Matrix dL_dv1(PICTURE_DIM / 2, PICTURE_DIM / 2);
        Matrix dL_dk2(KERNEL_DIM, KERNEL_DIM);
        Matrix dL_dv1_(PICTURE_DIM, PICTURE_DIM);
        Matrix dL_du1(PICTURE_DIM, PICTURE_DIM);
        Matrix dL_dk1(KERNEL_DIM, KERNEL_DIM);

        const int EARLY_STOPPING_PATIENCE = 3; // time to stop, when loss increase
        const std::string MODEL_FILE = "data/trained_model/model_weights.txt"; // file for weights
        const std::string BIN_MODEL_FILE = "data/trained_model/model_weights.bin"; // binary file for weights
        long double best_loss = std::numeric_limits<long double>::max();
        int epochs_no_improve = 0; // amount of epochs without reduction of the loss function

        const int batch_size = 64;
        const int num_samples = 60000;

        for (int ep = 0; ep < NUM_EPOCHS; ep++) {
            long double epoch_loss = 0;

            for (int i = 0; i < num_samples; i += batch_size) {
                long double batch_loss = 0;

                int actual_batch_size = std::min(batch_size, num_samples - i);

                // --- Average gradients to update weights ---
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
                    //First convolutional layer
                    U_1 = X.convolve(K_1); // convolve input data with kernel 1
                    V_1_ = U_1.ReLU(); // use activation function ReLU
                    std::tie(V_1, IND_1) = V_1_.max_pooling(POOLING_DIM, POOLING_DIM); // use maxpooling
                    // and write indices where was taken element during maxpooling for further unpooling

                    //Second convolutional layer
                    U_2 = V_1.convolve(K_2); // convolve previous results with kernel 2
                    V_2_ = U_2.ReLU(); // use activation function ReLU
                    std::tie(V_2, IND_2) = V_2_.max_pooling(POOLING_DIM, POOLING_DIM); // use maxpooling
                    // and store indices of this maxpooling

                    Matrix X_ = V_2.ravel(); // transform x from matrix (7x7) to vector (1x49)

                    //First fully-connected layer
                    T_1 = X_ * W_1 + b_1; // calculate output results of first layer
                    H_1 = T_1.ReLU(); // use activation function ReLU

                    //Second fully-connected layer
                    T_2 = H_1 * W_2 + b_2; // calculate output results of second layer
                    Z = T_2.softmax(); // use Softmax to distribute the probabilities of classes

                    long double L = Z.cross_entropy(Y); // calculate loss function for current example
                    batch_loss += L;

                    // --- Backward pass ---
                    // Calculate gradients of matrices with chain rule to updates parametrs
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

                    // --- Calculate sum of gradients of all parametrs
                    avg_dL_dw1 += dL_dw1;
                    avg_dL_db1 += dL_db1;
                    avg_dL_dw2 += dL_dw2;
                    avg_dL_db2 += dL_db2;
                    avg_dL_dk1 += dL_dk1;
                    avg_dL_dk2 += dL_dk2;
                }

                // --- Average all gradients on batch -- 
                avg_dL_dw1 /= actual_batch_size;
                avg_dL_db1 /= actual_batch_size;
                avg_dL_dw2 /= actual_batch_size;
                avg_dL_db2 /= actual_batch_size;
                avg_dL_dk1 /= actual_batch_size;
                avg_dL_dk2 /= actual_batch_size;

                // --- Update all weights ---
                W_1 -= avg_dL_dw1 * LEARNING_RATE;
                W_2 -= avg_dL_dw2 * LEARNING_RATE;
                b_1 -= avg_dL_db1 * LEARNING_RATE;
                b_2 -= avg_dL_db2 * LEARNING_RATE;
                K_1 -= avg_dL_dk1 * LEARNING_RATE;
                K_2 -= avg_dL_dk2 * LEARNING_RATE;

                epoch_loss += batch_loss;
            }

    epoch_loss /= (num_samples);

    // Create logs
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
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
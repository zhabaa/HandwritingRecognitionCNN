#include "../include/utilits.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <omp.h>

void readMNISTCSV(const std::string& filename, 
                  std::vector<Matrix>& images, 
                  std::vector<int>& labels) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    size_t line_count = 0;

    std::getline(file, line); // пропускаем заголовок

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
                img(i, j) = std::stod(value) / 255.0;
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

std::vector<Matrix> one_hot_encoding_labels(const std::vector<int>& labels) {
    std::vector<Matrix> one_hot_encoded_labels(labels.size());

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
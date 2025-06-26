#ifndef UTILITS_H
#define UTILITS_H

#include <string>
#include <vector>
#include "matrix.h"

#define CLASSES 10

void readMNISTCSV(const std::string& filename, 
                  std::vector<Matrix>& images, 
                  std::vector<int>& labels);

std::vector<Matrix> one_hot_encoding_labels(const std::vector<int>& labels);

void save_model(const Matrix& K1, const Matrix& K2,
                const Matrix& W1, const Matrix& W2,
                const Matrix& b1, const Matrix& b2,
                const std::string& filename);

#endif // UTILITS_H
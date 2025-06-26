g++ train_model.cpp src/matrix.cpp src/utilits.cpp -o train_model -fopenmp -lm
./train_model > data/trained_model/output.txt
rm train_model
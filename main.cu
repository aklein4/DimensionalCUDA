#include <iostream>
#include "dim_cuda.cuh"

int main() {
    size_t test_size = 128;

    // create a column vector and write values to it
    float* x_setter = new float[test_size];
    for (int i=0; i<test_size; i++) {
        x_setter[i] = i;
    }
    gpu::Vector1D<float> x(test_size);
    x.write(x_setter);
    delete[] x_setter;

    // create a 2D matrix and write values to it
    float* writer = new float;
    gpu::Matrix2D<float> A(test_size, test_size);
    for (int i=0; i<test_size; i++) {
        for (int j=0; j<test_size; j++) {
            *writer = i + j;
            A[j][i].write(writer);
        }
    }
    delete writer;

    // read the values into a buffer from y
    gpu::Vector1D<float> *y = &x;
    float* y_out = new float[test_size];
    y->read(y_out);

    // print the values of y_out buffer
    for (size_t i=0; i<test_size; i++) {
        std::cout << y_out[i] << std::endl;
    }
    delete[] y_out;

    return 0;
};
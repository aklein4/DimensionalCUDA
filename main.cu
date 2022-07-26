#include <iostream>

#include "gpu_types.cuh"
#include "gpu_math.cuh"
#include "timer.h"

int main() {
    size_t test_size = 32;

    // create a column vector and write values to it
    float* x_setter = new float[test_size];
    for (int i=0; i<test_size; i++) {
        x_setter[i] = static_cast<float>(i);
    }

    // create a 2D matrix and write values to it
    float* A_setter = new float[test_size * test_size];
    for (int i=0; i<test_size; i++) {
        for (int j=0; j<test_size; j++) {
            A_setter[i+(j*test_size)] = static_cast<float>(i + j);
        }
    }

    // create another column vector and write zero to all values
    float* y_setter = new float[test_size];
    for (int i=0; i<test_size; i++) {
        y_setter[i] = 0.0;
    }


    Timer initial_timer(TIME_UNIT::us);

    // create gpu data structures and fill them
    gpu::Vector1D<float> x(test_size);
    x.write(x_setter);
    gpu::Matrix2D<float> A(test_size, test_size);
    A.write(A_setter);
    gpu::Vector1D<float> y(test_size);
    y.write(y_setter);

    Timer timer(TIME_UNIT::us);

    // calculate Ax=y
    gpu::matMulti<float, float, float>(A, x, y, 0.0, false);

    timer.print("CUDA:");

    // read the values into a buffer from y
    float* y_out = new float[test_size];
    y.read(y_out);

    initial_timer.print("CUDA (with transfers):");


    timer.restart();
    // calculate the multiplication using the cpu
    for (int i=0; i<test_size; i++) {
        for (int j=0; j<test_size; j++) {
            y_setter[j] += A_setter[i+(j*test_size)] * x_setter[i];
        }
    }

    timer.print("CPU:");


   for (int i=0; i<test_size; i++) {
    std::cout << y_setter[i] << " -> " << y_out[i] << std::endl;
   }


    delete[] x_setter;
    delete[] A_setter;
    delete[] y_setter;
    delete[] y_out;

    return 0;
};
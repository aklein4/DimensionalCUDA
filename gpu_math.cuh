#ifndef GPU_MATH_H
#define GPU_MATH_H

#include <iostream>

namespace DIM_CUDA_HIDDEN {

/** Take matrix A and vector x and store the result to y.
 * NOTE: for * overloading purposes, the multiplication operation is B<V> = A<T> * x<U>
 * NOTE: matrix width/input size must be a multiple of 32
 * \param[in] A pointer to input matrix
 * \param[in] x pointer to input vector
 * \param[in] y pointer to output vector
 * \param[in] size width of the matrix/height of the input vector
 * \param[in] zero value to start summing into y from
 */
template <class T, class U, class V>
__global__ void gpu_matMulti(T* A, U* x, V* y, int size, V zero) {
    __shared__ V s[32];

    // i = location in input vector x
    int i = threadIdx.x;
    // j = location in output vector y
    int j = blockIdx.x;

    // sum up the convolutions between i and i+32
    V first_sum = zero;
    for (int n = 0; n < size/32; n++) {
        int ind = (j*size) + i + n;

        first_sum += A[ind] * x[i+n];
    }

    // put into shared memory for mutual access
    s[i] = first_sum;
    __syncthreads();

    #pragma unroll
    for (int w = 16; w > 1; w/=2) {
        if (i < w) {
            float my_sum = s[i];
            my_sum += s[i+w];
            s[i] = my_sum;
        }
        __syncthreads();
    }

    if (i == 0) {
        float my_sum = s[0];
        my_sum += s[1];
        y[j] = my_sum;
    }
};

}

namespace gpu {

/** Take matrix A and column vector x and store the product Ax into column vector y
 * NOTE: for overloading purposes, multiplication operations use [V = T * U]
 * NOTE: matrix width/input size must be a multiple of 32
 * \param[in] A input matrix of type T
 * \param[in] x input vector of type U
 * \param[in] y output matrix of type V
 * \param[in] zero value to start summing y values from
 * \param[in] check whether to check that sizes match up (default: true)
 */
template <class T, class U, class V>
void matMulti(Matrix2D<T> &A, Vector1D<U> &x, Vector1D<V> &y, V zero, bool check=true) {
    // check that sizes match up
    if (check) {
        if (A.width() != x.size()) std::cout << "ERROR: Matrix Multiplication input size does not match matrix!" << std::endl;
        if (A.width() % 32 != 0) std::cout << "ERROR: Matrix Multiplication input size is not multiple of 32!" << std::endl;
        if (A.height() != y.size()) std::cout << "ERROR: Matrix Multiplication input size does not match matrix!" << std::endl;
    }

    // do multiplication
    DIM_CUDA_HIDDEN::gpu_matMulti<T, U, V><<<A.height(), 32>>>(
        A.get_data(), x.get_data(), y.get_data(), A.width(), zero
    );
};

}

#endif
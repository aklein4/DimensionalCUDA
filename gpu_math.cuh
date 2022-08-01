#ifndef GPU_MATH_H
#define GPU_MATH_H

#include "gpu_types.cuh"

int MULT_BLOCK_SIZE = 8;
int SUM_BLOCK_SIZE = 32;
int SUM_DIVIDER = 8;

namespace DIM_CUDA_HIDDEN {

/* Take matrix A and vector x and store x[i]*A[j][i] to B[j][i].
 * NOTE: for * overloading purposes, the multiplication operation is B<V> = x<U> * A<T>
 * \param[in] A pointer to input matrix
 * \param[in] x pointer to input vector
 * \param[in] B pointer to cache matrix to store products
 * \param[in] height number of rows in A and B
 * \param[in] width number of columns in A and B, also number of elements in x
 */
template <class T, class U, class V>
__global__ void gpu_math_multiply_to_cache_matrix(T* A, U* x, V* B, int width, int height) {
    // get location
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (i >= width || j >= height) return;

    // store product
    int ind = i + (j * width);
    B[ind] = x[i] * A[ind];
};

/* Take matrix B and "squash" its rows into skinnier rows using summation.
 * constant SUM_DIVIDER determines the number of B columns that are summed into each C column.
 * NOTE: the width of C should be ceiling(width/SUM_DIVIDER)
 * \param[in] B pointer to input matrix
 * \param[in] C pointer to cache matrix to store sums
 * \param[in] height number of rows in B and C
 * \param[in] width number of columns B
 * \param[in] width_C number of columns in C
 * \param[in] sum_divider the number of B columns to sum into each C column
 * \param[in] zero Value to set an object to before summing
 */
template <class T>
__global__ void gpu_math_first_summation(T* B, T* C, int width, int height, int width_C, int sum_divider, T zero) {
    // get location
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (i >= width || j >= height) return;

    // get the index of each matrix
    int x = i * sum_divider;
    int ind_B = x + (j * width);
    int ind_C = i + (j * width_C);

    // sum the columns of B into C
    C[ind_C] = zero;
    for (int offset=0; offset < sum_divider && x + offset < width; offset++) {
        C[ind_C] += B[ind_B + offset];
    }
}

/* Sum the rows of matrix C into column vector y.
 * \param[in] C pointer to input matrix
 * \param[in] y pointer to output column vector
 * \param[in] height number of rows in C and elements in y
 * \param[in] width number of columns C
 * \param[in] zero Value to set an object to before summing
 */
template <class T>
__global__ void gpu_math_final_summation(T* C, T* y, int width, int height, T zero) {
    // get location
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (i != 0 || j >= height) return;

    // get the index of the first element of the row in C
    int ind = i + (j * width);

    // sum of the columns of C into y
    y[j] = zero;
    for (int offset=0; offset < width; offset++) {
        y[j] += C[ind + offset];
    }
}

}

namespace gpu {

/* Take matrix A and column vector x and store the product Ax into column vector y.
 * NOTE: for overloading purposes, multiplication operations use [V = U * T] and addition operations use [V = V + V]
 * \param[in] A input matrix of type T
 * \param[in] x input vector of type U
 * \param[in] y output matrix of type V
 * \param[in] height number of rows in A and B
 * \param[in] width number of columns in A and B, also number of elements in x
 * \param[in] zero value to set V object to before summing
 */
template <class T, class U, class V>
void matMulti(Matrix2D<T> &A, Vector1D<U> &x, Vector1D<V> &y, V zero) {
    // check that sizes match up
    assert(A.width() == x.size());
    assert(A.height() == y.size());

    // create a cache matrix to contain the multiplication stage output
    V* B;
    cudaError_t rval = cudaMalloc(&B, A.width() * A.height() * sizeof(V));
    assert(rval == cudaSuccess);

    // set block dimensions
    dim3 blockSize(MULT_BLOCK_SIZE, MULT_BLOCK_SIZE);

    // set block array for multiplication stage
    int num_width = A.width() / MULT_BLOCK_SIZE;
    if (A.width() % MULT_BLOCK_SIZE != 0) num_width++;
    int num_height = A.height() / MULT_BLOCK_SIZE;
    if (A.height() % MULT_BLOCK_SIZE != 0) num_height++;
    dim3 numBlocks(num_width, num_height);

    // Multiplication stage: Multiply each entry of A with corresponding x value
    DIM_CUDA_HIDDEN::gpu_math_multiply_to_cache_matrix<T, U, V><<<numBlocks, blockSize>>>(
        A.get_data(), x.get_data(), B, static_cast<int>(A.width()), static_cast<int>(A.height())
    );

    // create second cache matrix for first summation stage
    V* C;
    int width_C = A.width() / SUM_DIVIDER;
    if (A.width() % SUM_DIVIDER != 0) width_C++;
    rval = cudaMalloc(&C, A.height() * width_C * sizeof(V));
    assert(rval == cudaSuccess);

    // set block array for first summation stage
    num_width = width_C / MULT_BLOCK_SIZE;
    if (width_C % MULT_BLOCK_SIZE != 0) num_width++;
    numBlocks = dim3(num_width, num_height);

    // First summation stage: Collapse the rows of B into a skinnier matrix C
    DIM_CUDA_HIDDEN::gpu_math_first_summation<V><<<numBlocks, blockSize>>>(
        B, C, static_cast<int>(A.width()), static_cast<int>(A.height()), width_C, SUM_DIVIDER, zero
    );

    // set block params for final summation
    blockSize = dim3(1, SUM_BLOCK_SIZE);
    num_height = y.size() / SUM_BLOCK_SIZE;
    if (y.size() % SUM_BLOCK_SIZE != 0) num_height++;
    numBlocks = dim3(1, num_height);

    // Final Summation: Collapse the rows of C into the entries of y
    DIM_CUDA_HIDDEN::gpu_math_final_summation<V><<<numBlocks, blockSize>>>(
        C, y.get_data(), width_C, static_cast<int>(y.size()), zero
    );


    // free allocated memory
    cudaFree(B);
    cudaFree(C);

};

}

#endif
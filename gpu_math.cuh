#ifndef GPU_MATH_H
#define GPU_MATH_H

namespace DIM_CUDA_HIDDEN {

/* Take matrix A and vector x and store the result to y.
 * NOTE: for * overloading purposes, the multiplication operation is B<V> = x<U> * A<T>
 * NOTE: the maximum vector size/matrix width is 1024 (512 for compute capability 1.0).
 * \param[in] A pointer to input matrix
 * \param[in] x pointer to input vector
 * \param[in] y pointer to output vector
 */
template <class T, class U, class V>
__global__ void gpu_matMulti(T* A, U* x, V* y) {
    // get location
    int i = threadIdx.x;
    int j = blockIdx.y;

    // create shared memory to cache products
    extern __shared__ V B[];

    // store product
    B[i] = x[i] * A[i + (j * blockDim.x)];
    __syncthreads();

    // parrellel reduction of cache into vector
    for (int reduc = blockDim.x/2; reduc > 0; reduc /= 2) {
        if (i < reduc) B[i] += B[i+reduc];
        __syncthreads();
    }

    // set output
    if (i == 0) y[j] = B[0];
};

}

namespace gpu {

/* Take matrix A and column vector x and store the product Ax into column vector y
 * (Checks sizes)
 * NOTE: for overloading purposes, multiplication operations use [V = U * T]
 * \param[in] A input matrix of type T
 * \param[in] x input vector of type U
 * \param[in] y output matrix of type V
 */
template <class T, class U, class V>
void matMulti(Matrix2D<T> &A, Vector1D<U> &x, Vector1D<V> &y) {
    // check that sizes match up
    assert(A.width() == x.size());
    assert(A.height() == y.size());

    // set block dimensions
    dim3 blockSize = dim3(static_cast<int>(A.width()), 1); // wide
    dim3 numBlocks = dim3(1, static_cast<int>(x.size())); // tall

    // do multiplication
    DIM_CUDA_HIDDEN::gpu_matMulti<T, U, V><<<numBlocks, blockSize, blockSize.x * sizeof(V)>>>(
        A.get_data(), x.get_data(), y.get_data()
    );
};

/* Take matrix A and column vector x and store the product Ax into column vector y.
 * (DOES NOT CHECK SIZES)
 * NOTE: for overloading purposes, multiplication operations use [V = U * T]
 * \param[in] A input matrix of type T
 * \param[in] x input vector of type U
 * \param[in] y output matrix of type V
 */
template <class T, class U, class V>
void matMulti_opt(Matrix2D<T> &A, Vector1D<U> &x, Vector1D<V> &y) {
    // set block dimensions
    dim3 blockSize = dim3(static_cast<int>(A.width()), 1); // wide
    dim3 numBlocks = dim3(1, static_cast<int>(x.size())); // tall

    DIM_CUDA_HIDDEN::gpu_matMulti<T, U, V><<<numBlocks, blockSize, blockSize.x * sizeof(V)>>>(
        A.get_data(), x.get_data(), y.get_data()
    );
};

}

#endif
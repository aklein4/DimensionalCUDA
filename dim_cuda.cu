
#include "dim_cuda.h"

int MULT_BLOCK_SIZE = 8;
int SUM_DIVIDER = 8;

using namespace gpu;

template <class T>
Matrix3D<T>::Matrix3D(size_t num_layers, size_t column_h, size_t row_w, bool preprocess) {
    // set dimensions
    num_layers_ = num_layers;
    column_h_ = column_h;
    row_w_ = row_w;
    size_bytes = sizeof(T) * num_layers_ * column_h_ * row_w_;

    // data is always new allocated for 3D
    cudaError_t rval = cudaMalloc(&data_, size_bytes);
    assert(rval == cudaSuccess);

    // only process if needed
    if (preprocess) {
        process();
    }
    else {
        list_ = NULL;
    }
}

template <class T>
Matrix3D<T>::~Matrix3D() {
    // both were always new allocated
    if (list_ != NULL) delete[] list_;
    cudaFree(data_);
}

template <class T>
void Matrix3D<T>::process() {
    if (list_ != NULL) return;

    // create an array of 2D matrices to return if indexed
    list_ = new Matrix2D<T>[num_layers];
    for (int i=0; i < num_layers_; i++) {
        // each matrix should reference piece of memory
        list_[i] = Matrix2D<T>(column_h_, row_w_, data_+(i*column_h_*row_w_));
    }
}

template <class T>
void Matrix3D<T>::write(T* buf) {
    assert(buf != NULL);

    cudaError_t rval = cudaMemcpy(data_, buf, size_bytes, cudaMemcpyHostToDevice);
    assert(rval == cudaSuccess);
}

template <class T>
T* Matrix3D<T>::read(T* buf) {
    if (buf == NULL) {
        buf = new T[num_layers_ * column_h_ * row_w_];
    }

    cudaError_t rval = cudaMemcpy(buf, data_, size_bytes, cudaMemcpyDeviceToHost);
    assert(rval == cudaSuccess);
    return buf;
}


template <class T>
Matrix2D<T>::Matrix2D(size_t column_h, size_t row_w, T* data_loc, bool preprocess=true) {
    // set dimensions
    column_h_ = column_h;
    row_w_ = row_w;
    size_bytes = sizeof(T) * column_h_ * row_w_;

    // Allocate or reference data. Record if new allocated for later deletion.
    if (data_loc == NULL) {
        internal_ = false;
        cudaError_t rval = cudaMalloc(&data_, size_bytes);
        assert(rval == cudaSuccess);
    }
    else {
        internal_ = true;
        data_ = data_loc;
    }

    // only process if needed
    if (preprocess) {
        process();
    }
    else {
        layer_ = NULL;
    }
}

template <class T>
Matrix2D<T>::~Matrix2D() {
    // always new allocated
    if (layer_ != NULL) delete[] layer_;
    // only delete if originally new allocated
    if (!internal_) cudaFree(data_);
}

template <class T>
void Matrix2D<T>::process() {
    if (layer_ != NULL) return;

    // create an array of 2D matrices to return if indexed
    layer_ = new Vector1D<T>[column_h_];
    for (int i=0; i < row_w_; i++) {
        // each matrix should reference piece of memory
        layer_[i] = Vector1D<T>(row_w_, data_+(i*row_w_));
    }
}

template <class T>
void Matrix2D<T>::write(T* buf) {
    assert(buf != NULL);

    cudaError_t rval = cudaMemcpy(data_, buf, size_bytes, cudaMemcpyHostToDevice);
    assert(rval == cudaSuccess);
}

template <class T>
T* Matrix2D<T>::read(T* buf) {
    if (buf == NULL) {
        buf = new T[row_w_ * column_h_];
    }

    cudaError_t rval = cudaMemcpy(buf, data_, size_bytes, cudaMemcpyDeviceToHost);
    assert(rval == cudaSuccess);
    return buf;
}


template <class T>
Matrix1D<T>::Matrix1D(size_t row_w, T* data_loc, bool preprocess=true) {
    // set dimensions
    row_w_ = row_w;
    size_bytes = sizeof(T) * row_w_;

    // Allocate or reference data. Record if new allocated for later deletion.
    if (data_loc == NULL) {
        internal_ = false;
        cudaError_t rval = cudaMalloc(&data_, size_bytes);
        assert(rval == cudaSuccess);
    }
    else {
        internal_ = true;
        data_ = data_loc;
    }

    // only process if needed
    if (preprocess) {
        process();
    }
    else {
        ptrs_ = NULL;
    }
}

template <class T>
Matrix1D<T>::~Matrix1D() {
    if (ptrs_ != NULL) delete[] ptrs_;
    // only delete if originally new allocated
    if (!internal_) cudaFree(data_);
}

template <class T>
void Matrix1D<T>::process() {
    if (ptrs_ != NULL) return;

    // create an array of 2D matrices to return if indexed
    ptrs_ = new device_ptr<T>[row_w_];
    for (int i=0; i < row_w_; i++) {
        // each matrix should reference piece of memory
        ptrs_[i] = device_ptr<T>(data_+i);
    }
}

template <class T>
void Matrix1D<T>::write(T* buf) {
    assert(buf != NULL);

    cudaError_t rval = cudaMemcpy(data_, buf, size_bytes, cudaMemcpyHostToDevice);
    assert(rval == cudaSuccess);
}

template <class T>
T* Matrix1D<T>::read(T* buf) {
    if (buf == NULL) {
        buf = new T[row_w_];
    }

    cudaError_t rval = cudaMemcpy(buf, data_, size_bytes, cudaMemcpyDeviceToHost);
    assert(rval == cudaSuccess);
    return buf;
}


template <class T>
Vector1D<T>::Vector1D(size_t size, bool preprocess=true) {
    // set dimensions
    size_ = size;
    size_bytes = sizeof(T) * size_;

    // Allocate data
    cudaError_t rval = cudaMalloc(&data_, size_bytes);
    assert(rval == cudaSuccess);

    // only process if needed
    if (preprocess) {
        process();
    }
    else {
        ptrs_ = NULL;
    }
}

template <class T>
Vector1D<T>::~Vector1D() {
    if (ptrs_ != NULL) delete[] ptrs_;
    cudaFree(data_);
}

template <class T>
void Vector1D<T>::process() {
    if (ptrs_ != NULL) return;

    // create an array of 2D matrices to return if indexed
    ptrs_ = new device_ptr<T>[size_];
    for (int i=0; i < row_w_; i++) {
        // each matrix should reference piece of memory
        ptrs_[i] = device_ptr<T>(data_+i);
    }
}

template <class T>
void Vector1D<T>::write(T* buf) {
    assert(buf != NULL);

    cudaError_t rval = cudaMemcpy(data_, buf, size_bytes, cudaMemcpyHostToDevice);
    assert(rval == cudaSuccess);
}

template <class T>
T* Vector1D<T>::read(T* buf) {
    if (buf == NULL) {
        buf = new T[row_w_];
    }

    cudaError_t rval = cudaMemcpy(buf, data_, size_bytes, cudaMemcpyDeviceToHost);
    assert(rval == cudaSuccess);
    return buf;
}


template <class T>
device_ptr<T>::device_ptr(T* data_loc) {
    size_bytes = sizeof(T);

    // Allocate or reference data. Record if new allocated for later deletion.
    if (data_loc == NULL) {
        internal_ = false;
        cudaError_t rval = cudaMalloc(&data_, size_bytes);
        assert(rval == cudaSuccess);
    }
    else {
        internal_ = true;
        data_ = data_loc;
    }
}

template <class T>
device_ptr<T>::~device_ptr() {
    // only delete if originally new allocated
    if (!internal_) cudaFree(data_);
}

template <class T>
void device_ptr<T>::write(T* buf) {
    assert(buf != NULL);

    cudaError_t rval = cudaMemcpy(data_, buf, size_bytes, cudaMemcpyHostToDevice);
    assert(rval == cudaSuccess);
}

template <class T>
T* device_ptr<T>::read(T* buf) {
    if (buf == NULL) {
        buf = new T;
    }

    cudaError_t rval = cudaMemcpy(buf, data_, size_bytes, cudaMemcpyDeviceToHost);
    assert(rval == cudaSuccess);
    return buf;
}

template <class T, class V>
__global__ void multiply_to_cache_matrix(T* A, V* x, V* B, int height, int width) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (i >= width || j >= height) return;

    int ind = i + (j * width);
    B[ind] = x[i] * A[ind];
}

template <class V>
__global__ void first_summation(V* B, V* C, int height, int width, int num_cells) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (i >= width || j >= height) return;

    int count = SUM_DIVIDER;
    if ((i + 1) * SUM_DIVIDER >= width) count = width - (i * SUM_DIVIDER);

    int ind_B = (i * SUM_DIVIDER) + (j * width);
    int ind_C = i + (j * num_cells);
    C[ind_C] = 0;
    for (int offset=0; offset < SUM_DIVIDER; offset++) {
        C[ind_C] += B[ind_B + offset];
    }
}

template <class V>
__global__ void final_summation(V* C, V* y, int height, int num_cells) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (i != 0 || j >= height) return;

    V sum = 0;
    for (int offset=0; offset<num_cells; offset++) {
        sum += C[offset + (j * height)];
    }
    y[j] = sum;
}



template <class T, class V>
Vector1D<V>* matMulti(const Matrix2D<T> &A, const Vector1D<V> &x, Vector1D<V>* y) {
    // check that sizes match up
    assert(A.width() == x.size());
    if (y != NULL) assert (A.height() == y->size());

    // if y not provided, create it
    if (y == NULL) {
        y = new Vector1D<V>;
        y[0] = Vector1D<V>(A.height(), false);
    }

    // create a cache matrix to contain the multiplication stage output
    V* B;
    cudaMalloc(&B, A.width() * A.height() * sizeof(V));

    // set gpu dimensions for multiplication stage
    dim3 blockSize(MULT_BLOCK_SIZE, MULT_BLOCK_SIZE);

    int num_width = A.width() / MULT_BLOCK_SIZE;
    if (A.width() % MULT_BLOCK_SIZE != 0) num_width++;

    int num_height = A.height() / MULT_BLOCK_SIZE;
    if (A.height() % MULT_BLOCK_SIZE != 0) num_width++;

    dim3 numBlocks(num_width, num_height);

    // Multiplication stage: Multiply each entry of A with corresponding x value
    multiply_to_cache_matrix<<<numBlocks, blockSize>>>(A.data_, x.data_, B, A.width(), A.height());
 
    // create second cache matrix for first summation stage
    int num_cells = A.width() / SUM_DIVIDER;
    if (A.width() % SUM_DIVIDER != 0) num_width++;
    V* C;
    cudaMalloc(&C, num_cells * A.height() * sizeof(V));

    // set gpu dimensions for summation stage
    num_width = num_cells / MULT_BLOCK_SIZE;
    numBlocks = dim3(num_width, num_height);

    // First summation stage: Collapse the rows of B into a skinnier matrix
    first_summation<<<numBlocks, blockSize>>>(B, C, A.width(), A.height(), num_cells);

    // Final Summation: Collapse the rows of C into the entries of y
    blockSize = dim3(1, y->size());
    final_summation<<<1, blockSize>>>(C, y, y->size(), num_cells);

    // free allocated memory
    cudaFree(B);
    cudaFree(C);

    return y;
}
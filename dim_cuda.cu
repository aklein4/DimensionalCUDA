
#include "dim_cuda.h"

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
    // create an array of 2D matrices to return if indexed
    layer_ = new Matrix1D<T>[column_h_];
    for (int i=0; i < row_w_; i++) {
        // each matrix should reference piece of memory
        layer_[i] = Matrix1D<T>(row_w_, data_+(i*row_w_));
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
    // create an array of 2D matrices to return if indexed
    ptrs_ = new device_ptr<T>[row_w_];
    for (int i=0; i < row_w_; i++) {
        // each matrix should reference piece of memory
        ptrs_[i] = device_ptr<T>();
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
#ifndef DIM_CUDA_H
#define DIM_CUDA_H

#include <iostream>
#include <assert.h>
#include <vector>

namespace gpu {

// prototype data sub-structures and math functions
template <class T> class Matrix2D;
template <class T> class Matrix1D;
template <class T> class Vector1D;
template <class T> class device_ptr;
template <class T, class V> Vector1D<V>* matMulti(const Matrix2D<T> &A, const Vector1D<V> &x, Vector1D<V>* y=NULL);

/* A 3D matrix, which is essentially a list of 2D matrixes.
 * Data is stored contiguously in device memory, and indexing returns a 
 * 2D matrix whose data is a section of that memory.
 */
template <class T> 
class Matrix3D {
    public:
        /* Create a 3D matrix of the specified size.
         * Data will be new allocated in device memory and deleted upon deconstruction.
         * \param[in] num_layers Number of 2D matrix layers
         * \param[in] column_h Height of a child 2D matrix's column
         * \param[in] row_w Width of a child 2D matrix's row
         */
        Matrix3D(size_t num_layers, size_t column_h, size_t row_w) {
                // set dimensions
                num_layers_ = num_layers;
                column_h_ = column_h;
                row_w_ = row_w;
                size_bytes = sizeof(T) * num_layers_ * column_h_ * row_w_;

                // data is always new allocated for 3D
                cudaError_t rval = cudaMalloc(&data_, size_bytes);
                assert(rval == cudaSuccess);
        };

        /* Data is deleted upon deconstruction, and any child matrices
         * will index to invalid data. */
        ~Matrix3D() {
            // both always new allocated
            cudaFree(data_);
        };

        /* Return a 2D matrix which references a piece of the contiguous memory.
         * If a list of sub-matrices hasn't been created, it is created.
         */
        Matrix2D<T>& operator [](size_t i) {
            assert(i < num_layers_);

            if (list_.size() == 0) process();

            return list_[i];
        };

        /* read data from the device.
         * If no buffer given, one will be new allocated and must be freed by caller.
         * \param[in] buf Buffer to write memory into. */
        void write(T* buf){
            assert(buf != NULL);

            cudaError_t rval = cudaMemcpy(data_, buf, size_bytes, cudaMemcpyHostToDevice);
            assert(rval == cudaSuccess);
        };

        /* Write the contents of the buffer into device memory.
         * \param[in] buf Buffer to copy from.
         */
        T* read(T* buf=NULL) {
            if (buf == NULL) {
                buf = new T[num_layers_ * column_h_ * row_w_];
            }

            cudaError_t rval = cudaMemcpy(buf, data_, size_bytes, cudaMemcpyDeviceToHost);
            assert(rval == cudaSuccess);
            return buf;
        };
    
        /* \return Number of 2D sub-matrices */
        size_t num() {return num_layers_; };
        /* \return Height of column */
        size_t height() {return column_h_; };
        /* \return Width of row */
        size_t width() {return row_w_; };

    private:
        // dimensions
        size_t num_layers_;
        size_t column_h_;
        size_t row_w_;

        // size of the array in bytes
        size_t size_bytes;

        // pointer to memory chunk containing data
        T* data_;

        // an array to hold sub-matrices
        std::vector<Matrix2D<T>> list_;

        // create sub-data array
        void process(){
            if (list_.size() != 0) return;

            for (int i=0; i < num_layers_; i++) {
                // each matrix should reference piece of memory
                list_.push_back(Matrix2D<T>(column_h_, row_w_, data_+(i*column_h_*row_w_)));
            }
        };
};


/* A 2D matrix, which is essentially a list of 1D matrixes.
 * Data is stored contiguously in memory, and indexing returns a 
 * 1D matrix whose data is a section of that memory.
 */
template <class T> 
class Matrix2D {
    public:
        /* Create a 2D matrix of the specified size.
         * If data_loc is not NULL, then the data will be set to 
         * that location and will not be deleted upon deconstuction.
         * \param[in] column_h Height of a child 2D matrix's column
         * \param[in] row_w Width of a child 2D matrix's row
         * \param[in] data_loc Pointer to the data to reference. Default NULL means new allocation.     
         */
        Matrix2D(size_t column_h, size_t row_w, T* data_loc=NULL) {
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
        };

        /* Deconstuction only deletes data if it was created in the constructor.
         * If that happens, then child 1D matrixes will index to invalid memory.*/
        ~Matrix2D() {
            // only delete if originally new allocated
            if (!internal_) cudaFree(data_);
        };
        
        /* Return a 1D matrix which references a piece of the contiguous memory.
         * If a sub-matrix vector has not been created, it is created.
         */
        Matrix1D<T>& operator [](size_t i) {
            assert(i < column_h_);

            if (layer_.size() == 0) process();

            return layer_[i];
        };

        /* read data from the device.
         * If no buffer given, one will be new allocated and must be freed by caller.
         * \param[in] buf Buffer to write memory into. */
        void write(T* buf)  {
            assert(buf != NULL);

            cudaError_t rval = cudaMemcpy(&data_, buf, size_bytes, cudaMemcpyHostToDevice);
            assert(rval == cudaSuccess);
        };

        /* Write the contents of the buffer into device memory.
         * \param[in] buf Buffer to copy from.
         */
        T* read(T* buf=NULL)  {
            if (buf == NULL) {
                buf = new T[row_w_ * column_h_];
            }

            cudaError_t rval = cudaMemcpy(buf, data_, size_bytes, cudaMemcpyDeviceToHost);
            assert(rval == cudaSuccess);

            return buf;
        };

        /* \return Height of column */
        size_t height() {return column_h_; };
        /* \return Width of row */
        size_t width() {return row_w_; };

        /* Friend Functions */
        template <class T, class V> Vector1D<T>* matMulti(const Matrix2D<T> &A, const Vector1D<V> &x, Vector1D<V>* y);

    private:
        // dimensions
        size_t column_h_;
        size_t row_w_;

        // size of the array in bytes
        size_t size_bytes;

        // pointer to memory chunk containing data
        T* data_;

        // whether this is a sub-matrix, if not then data must be deleted
        bool internal_;

        // an array to hold sub-matrices
        std::vector<Matrix1D<T>> layer_;

        // create sub-data array
        void process() {
            if (layer_.size() != 0) return;
            // create an array of 2D matrices to return if indexed
            for (int i=0; i < row_w_; i++) {
                layer_.push_back(Matrix1D<T>(row_w_, data_));
            }
        };
};


/* A 1D row matrix (transpose of vector), whose data is stored contigously in device memory.
 * Indexing returns a reference to an entry in that memory.
 */
template <class T>
class Matrix1D {
    public:
        /* Create a 1D matrix of the specified size.
         * If data_loc is not NULL, then the data will be set to 
         * that location and will not be deleted upon deconstuction.
         * \param[in] row_w Width of a child 2D matrix's row
         * \param[in] data_loc Pointer to the data to reference. Default NULL means new allocation.
         */
        Matrix1D(size_t row_w, T* data_loc=NULL) {
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
        };

        /* Deconstuction only deletes data if it was created in the constructor */
        ~Matrix1D()  {
            // only delete if originally new allocated
            if (!internal_) cudaFree(data_);
        };

        /* Return a reference to a piece of the contiguous memory.
         */
        device_ptr<T>& operator [](size_t i) {
            assert(i < row_w_);

            if (ptrs_.size() == 0) process();

            return ptrs_[i];
        };

        /* read data from the device.
         * If no buffer given, one will be new allocated and must be freed by caller.
         * \param[in] buf Buffer to write memory into. */
        void write(T* buf) {
            assert(buf != NULL);

            cudaError_t rval = cudaMemcpy(data_, buf, size_bytes, cudaMemcpyHostToDevice);
            assert(rval == cudaSuccess);
        };

        /* Write the contents of the buffer into device memory.
         * \param[in] buf Buffer to copy from.
         */
        T* read(T* buf=NULL) {
            if (buf == NULL) {
                buf = new T[row_w_];
            }

            cudaError_t rval = cudaMemcpy(buf, data_, size_bytes, cudaMemcpyDeviceToHost);
            assert(rval == cudaSuccess);
            return buf;
        };

        /* \return Width of row */
        size_t width() {return row_w_; };

    private:
        // dimensions
        size_t row_w_;

        // size of the array in bytes
        size_t size_bytes;

        // pointer to memory chunk containing data
        T* data_;

        // whether this is a sub-matrix, if not then data must be deleted
        bool internal_;

        // an array to hold the device pointers to memory
        std::vector<device_ptr<T>> ptrs_;

        // create sub-data array
        void process() {
            if (ptrs_.size() != 0) return;

            for (int i=0; i < row_w_; i++) {
                ptrs_.push_back(device_ptr<T>(data_+i));
            }
        };
};


/* A 1D column vector, whose data is stored contigously in device memory.
 * Indexing returns a reference to an entry in that memory.
 */
template <class T>
class Vector1D {
    public:
        /* Create a 1D matrix of the specified size.
         * Data will be allocated and deleted upon deconstuction.
         * \param[in] size Length of the vector
         */
        Vector1D(size_t size) {
            // set dimensions
            size_ = size;
            size_bytes = sizeof(T) * size_;

            // Allocate data
            cudaError_t rval = cudaMalloc(&data_, size_bytes);
            assert(rval == cudaSuccess);
        };

        /* Deconstuction only deletes data if it was created in the constructor */
        ~Vector1D() {
            cudaFree(data_);
        };

        /* Return a reference to a piece of the contiguous memory.
         */
        device_ptr<T>& operator [](size_t i) {
            assert(i < size_);

            if (ptrs_.size() != 0) process();

            return ptrs_[i];
        };

        /* read data from the device.
         * If no buffer given, one will be new allocated and must be freed by caller.
         * \param[in] buf Buffer to write memory into. */
        void write(T* buf) {
            assert(buf != NULL);

            cudaError_t rval = cudaMemcpy(data_, buf, size_bytes, cudaMemcpyHostToDevice);
            assert(rval == cudaSuccess);
        };

        /* Write the contents of the buffer into device memory.
         * \param[in] buf Buffer to copy from.
         */
        T* read(T* buf=NULL) {
            if (buf == NULL) {
                buf = new T[size_];
            }

            cudaError_t rval = cudaMemcpy(buf, data_, size_bytes, cudaMemcpyDeviceToHost);
            assert(rval == cudaSuccess);
            return buf;
        };

        /* \return Width of row */
        size_t size() {return size_; };

        /* Friend Functions */
        template <class T, class V> Vector1D<T>* matMulti(const Matrix2D<T> &A, const Vector1D<V> &x, Vector1D<V>* y);

    private:
        // dimensions
        size_t size_;

        // size of the array in bytes
        size_t size_bytes;
        
        // pointer to memory chunk containing data
        T* data_;

        // an array to hold the device pointers to memory
        std::vector<device_ptr<T>> ptrs_;

        // create sub-data array
        void process() {
            if (ptrs_.size() != NULL) return;

            for (int i=0; i < size_; i++) {
                ptrs_.push_back(device_ptr<T>(data_+i));
            }
        };
};

/* A pointer to an object in device memory.
 */
template <class T>
class device_ptr {
    public:
        device_ptr(T* data_loc=NULL) {
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
        };

        ~device_ptr() {
            // only delete if originally new allocated
            if (!internal_) cudaFree(data_);
        };

        /* read data from the device.
         * If no buffer given, one will be new allocated and must be freed by caller.
         * \param[in] buf Buffer to write memory into. */
        void write(T* buf) {
            assert(buf != NULL);

            cudaError_t rval = cudaMemcpy(data_, buf, size_bytes, cudaMemcpyHostToDevice);
            assert(rval == cudaSuccess);
        }
        /* Write the contents of the buffer into device memory.
         * \param[in] buf Buffer to copy from.
         */
        T* read(T* buf=NULL) {
            if (buf == NULL) {
                buf = new T;
            }

            cudaError_t rval = cudaMemcpy(buf, data_, size_bytes, cudaMemcpyDeviceToHost);
            assert(rval == cudaSuccess);
            return buf;
        }

    private:
        // size of the object in bytes
        size_t size_bytes;

        // pointer to the object in device memory
        T* data_;

        // whether this is a sub-object, if not then data must be deleted
        bool internal_;
};

}

#endif
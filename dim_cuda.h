#ifndef LAYER_H
#define LAYER_H

#include <assert.h>

namespace gpu {

// prototype data sub-structures
template <class T> class Matrix2D;
template <class T> class Matrix1D;
template <class T> class device_ptr;

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
         * \param[in] preprocess Create indexable objects at contruction, otherwise at first index. Default: true
         */
        Matrix3D(size_t num_layers, size_t column_h, size_t row_w, bool preprocess=true);
        /* Data is deleted upon deconstruction, and any child matrices
         * will index to invalid data. */
        ~Matrix3D();

        /* Return a 2D matrix which references a piece of the contiguous memory.
         */
        Matrix2D<T>& operator [](size_t i) const {
            assert(i < num_layers_);
            if (list_ == NULL) process();
            return list_[i];
        };

        /* read data from the device.
         * If no buffer given, one will be new allocated and must be freed by caller.
         * \param[in] buf Buffer to write memory into. */
        void write(T* buf);
        /* Write the contents of the buffer into device memory.
         * \param[in] buf Buffer to copy from.
         */
        T* read(T* buf=NULL);

    
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
        Matrix2D<T>* list_;
        // create sub-data array
        void process();
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
         * \param[in] preprocess Create indexable objects at contruction, otherwise at first index. Default: true         
         */
        Matrix2D(size_t column_h, size_t row_w, T* data_loc=NULL, bool preprocess=true);
        /* Deconstuction only deletes data if it was created in the constructor.
         * If that happens, then child 1D matrixes will index to invalid memory.*/
        ~Matrix2D();
        
        /* Return a 1D matrix which references a piece of the contiguous memory.
         */
        Matrix1D<T>& operator [](size_t i) const {
            assert(i < column_h_);
            if (layer_ == NULL) process();
            return layer_[i];
        };

        /* read data from the device.
         * If no buffer given, one will be new allocated and must be freed by caller.
         * \param[in] buf Buffer to write memory into. */
        void write(T* buf);
        /* Write the contents of the buffer into device memory.
         * \param[in] buf Buffer to copy from.
         */
        T* read(T* buf=NULL);

        /* \return Height of column */
        size_t height() {return column_h_; };
        /* \return Width of row */
        size_t width() {return row_w_; };

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
        Matrix1D<T>* layer_;
        // create sub-data array
        void process();
};

/* A 1D matrix (transpose of vector), whose data is stored contigously in memory.
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
         * \param[in] preprocess Create indexable objects at contruction, otherwise at first index. Default: true
         */
        Matrix1D(size_t row_w, T* data_loc=NULL, bool preprocess=true);
        /* Deconstuction only deletes data if it was created in the constructor */
        ~Matrix1D();

        /* Return a reference to a piece of the contiguous memory.
         */
        T& operator [](size_t i) const {
            assert(i < row_w_);
            if (ptrs_ == NULL) process();
            return ptrs_[i];
        };

        /* read data from the device.
         * If no buffer given, one will be new allocated and must be freed by caller.
         * \param[in] buf Buffer to write memory into. */
        void write(T* buf);
        /* Write the contents of the buffer into device memory.
         * \param[in] buf Buffer to copy from.
         */
        T* read(T* buf=NULL);

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
        device_ptr* ptrs_;
        // create sub-data array
        void process();
};

/* A pointer to an object in device memory.
 */
template <class T>
class device_ptr {
    public:
        device_ptr(T* data_loc=NULL);
        ~device_ptr();

        /* read data from the device.
         * If no buffer given, one will be new allocated and must be freed by caller.
         * \param[in] buf Buffer to write memory into. */
        void write(T* buf);
        /* Write the contents of the buffer into device memory.
         * \param[in] buf Buffer to copy from.
         */
        T* read(T* buf=NULL);

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
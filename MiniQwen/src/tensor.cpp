#include "tensor.h"

#include <iostream>
#include <stdexcept>
#include <numeric>
#include <limits>


// ------------------------------------
// Helper: Calculate number of elements
// ------------------------------------

static size_t calculate_numel(
    const std::vector<size_t>& shape
) {
    if (shape.empty()) {
        throw std::invalid_argument(
            "Tensor shape cannot be empty"
        );
    }

    size_t total = 1;

    for (size_t dimension : shape) {

        if (dimension == 0) {
            throw std::invalid_argument(
                "Tensor dimensions cannot be zero"
            );
        }

        if (total >
            std::numeric_limits<size_t>::max()
            / dimension) {

            throw std::overflow_error(
                "Tensor size overflow"
            );
        }

        total *= dimension;
    }

    return total;
}


// ------------------------------------
// Constructors
// ------------------------------------

Tensor::Tensor(
    const std::vector<size_t>& shape
)
    : shape_(shape)
{
    size_t total = calculate_numel(shape_);

    data_.resize(total, 0.0f);

    calculate_strides();
}


Tensor::Tensor(
    const std::vector<size_t>& shape,
    const std::vector<float>& values
)
    : shape_(shape)
{
    size_t total = calculate_numel(shape_);

    if (total != values.size()) {
        throw std::invalid_argument(
            "Shape and data size mismatch"
        );
    }

    data_ = values;

    calculate_strides();
}


// ------------------------------------
// Stride calculation
// ------------------------------------

void Tensor::calculate_strides() {

    strides_.resize(shape_.size());

    size_t current_stride = 1;

    for (int i = static_cast<int>(shape_.size()) - 1;
         i >= 0;
         --i) {

        strides_[i] = current_stride;

        current_stride *= shape_[i];
    }
}


// ------------------------------------
// Metadata
// ------------------------------------

size_t Tensor::rank() const {
    return shape_.size();
}


size_t Tensor::size() const {
    return data_.size();
}


const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}


const std::vector<size_t>& Tensor::strides() const {
    return strides_;
}


// ------------------------------------
// Raw data access
// ------------------------------------

float* Tensor::data() {
    return data_.data();
}


const float* Tensor::data() const {
    return data_.data();
}


// ------------------------------------
// Multidimensional offset
// ------------------------------------

size_t Tensor::calculate_offset(
    const std::vector<size_t>& indices
) const {

    if (indices.size() != rank()) {
        throw std::invalid_argument(
            "Incorrect number of indices"
        );
    }

    size_t offset = 0;

    for (size_t i = 0; i < indices.size(); ++i) {

        if (indices[i] >= shape_[i]) {
            throw std::out_of_range(
                "Tensor index out of bounds"
            );
        }

        offset += indices[i] * strides_[i];
    }

    return offset;
}


// ------------------------------------
// Tensor access
// ------------------------------------

float& Tensor::at(
    std::initializer_list<size_t> indices
) {

    std::vector<size_t> idx(
        indices.begin(),
        indices.end()
    );

    return data_[calculate_offset(idx)];
}


const float& Tensor::at(
    std::initializer_list<size_t> indices
) const {

    std::vector<size_t> idx(
        indices.begin(),
        indices.end()
    );

    return data_[calculate_offset(idx)];
}


// ------------------------------------
// Flat access
// ------------------------------------

float& Tensor::operator[](size_t index) {

    if (index >= size()) {
        throw std::out_of_range(
            "Flat tensor index out of bounds"
        );
    }

    return data_[index];
}


const float& Tensor::operator[](size_t index) const {

    if (index >= size()) {
        throw std::out_of_range(
            "Flat tensor index out of bounds"
        );
    }

    return data_[index];
}


// ------------------------------------
// Print tensor
// ------------------------------------

void Tensor::print() const {

    std::cout << "Shape: [";

    for (size_t i = 0; i < rank(); ++i) {

        std::cout << shape_[i];

        if (i + 1 < rank()) {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";

    std::cout << "Strides: [";

    for (size_t i = 0; i < rank(); ++i) {

        std::cout << strides_[i];

        if (i + 1 < rank()) {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";

    std::cout << "Data: [";

    for (size_t i = 0; i < size(); ++i) {

        std::cout << data_[i];

        if (i + 1 < size()) {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";
}


// ====================================
// MATHEMATICAL OPERATIONS
// ====================================


// ------------------------------------
// Elementwise addition
// ------------------------------------

Tensor add(
    const Tensor& A,
    const Tensor& B
) {

    if (A.shape() != B.shape()) {
        throw std::invalid_argument(
            "Addition shape mismatch"
        );
    }

    Tensor result(A.shape());

    for (size_t i = 0; i < A.size(); ++i) {

        result[i] = A[i] + B[i];
    }

    return result;
}


// ------------------------------------
// Elementwise multiplication
// ------------------------------------

Tensor multiply(
    const Tensor& A,
    const Tensor& B
) {

    if (A.shape() != B.shape()) {
        throw std::invalid_argument(
            "Multiplication shape mismatch"
        );
    }

    Tensor result(A.shape());

    for (size_t i = 0; i < A.size(); ++i) {

        result[i] = A[i] * B[i];
    }

    return result;
}


// ------------------------------------
// Matrix transpose
// ------------------------------------

Tensor transpose(
    const Tensor& A
) {

    if (A.rank() != 2) {
        throw std::invalid_argument(
            "Transpose requires a 2D tensor"
        );
    }

    size_t rows = A.shape()[0];
    size_t cols = A.shape()[1];

    Tensor result({cols, rows});

    for (size_t i = 0; i < rows; ++i) {

        for (size_t j = 0; j < cols; ++j) {

            result.at({j, i}) = A.at({i, j});
        }
    }

    return result;
}


// ------------------------------------
// Matrix multiplication
// ------------------------------------

Tensor matmul(
    const Tensor& A,
    const Tensor& B
) {

    if (A.rank() != 2 || B.rank() != 2) {
        throw std::invalid_argument(
            "Matrix multiplication requires 2D tensors"
        );
    }

    size_t M = A.shape()[0];
    size_t K = A.shape()[1];

    size_t K2 = B.shape()[0];
    size_t N = B.shape()[1];

    if (K != K2) {
        throw std::invalid_argument(
            "Matrix multiplication dimension mismatch"
        );
    }

    Tensor C({M, N});

    for (size_t i = 0; i < M; ++i) {

        for (size_t j = 0; j < N; ++j) {

            float accumulator = 0.0f;

            for (size_t k = 0; k < K; ++k) {

                accumulator +=
                    A.at({i, k}) *
                    B.at({k, j});
            }

            C.at({i, j}) = accumulator;
        }
    }

    return C;
}


// ------------------------------------
// Sum reduction
// ------------------------------------

float sum(const Tensor& A) {

    float result = 0.0f;

    for (size_t i = 0; i < A.size(); ++i) {
        result += A[i];
    }

    return result;
}


// ------------------------------------
// Mean reduction
// ------------------------------------

float mean(const Tensor& A) {

    return sum(A) / static_cast<float>(A.size());
}
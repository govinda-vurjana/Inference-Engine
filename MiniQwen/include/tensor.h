#pragma once

#include <vector>
#include <initializer_list>
#include <cstddef>

class Tensor {
private:
    std::vector<float> data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;

    void calculate_strides();

    size_t calculate_offset(
        const std::vector<size_t>& indices
    ) const;

public:
    // Construct a tensor using its shape
    explicit Tensor(
        const std::vector<size_t>& shape
    );

    // Construct a tensor using shape and values
    Tensor(
        const std::vector<size_t>& shape,
        const std::vector<float>& values
    );

    // Basic metadata
    size_t rank() const;
    size_t size() const;

    const std::vector<size_t>& shape() const;
    const std::vector<size_t>& strides() const;

    // Access underlying memory
    float* data();
    const float* data() const;

    // Multidimensional access
    float& at(
        std::initializer_list<size_t> indices
    );

    const float& at(
        std::initializer_list<size_t> indices
    ) const;

    // Flat access
    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    // Display tensor
    void print() const;
};


// Mathematical operations, only declarations no definitions yet

Tensor add(
    const Tensor& A,
    const Tensor& B
);

Tensor multiply(
    const Tensor& A,
    const Tensor& B
);

Tensor transpose(
    const Tensor& A
);

Tensor matmul(
    const Tensor& A,
    const Tensor& B
);

float sum(const Tensor& A);

float mean(const Tensor& A); 
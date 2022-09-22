//
// Created by salyd on 18/02/2021.
//

#ifndef AUFILEPROCESSING_FFT_H
#define AUFILEPROCESSING_FFT_H

#include <vector>
#include <complex>
#include <array>
#include <numbers>

template<typename floating_type, std::size_t DIM>
constexpr std::array<std::complex<floating_type>, DIM / 2> twiddle_factors() {
    std::array<std::complex<floating_type>, DIM / 2> t;
    for (std::size_t k = 0; k < DIM / 2; k++)
        t[k] = std::exp(std::complex<floating_type>(0, -2 * std::numbers::pi * k / DIM));
    return t;
}

template<std::size_t DIM>
constexpr std::array<std::size_t, DIM> bit_reverse_array() {
    std::array<std::size_t, DIM> scrambled{};
    for (std::size_t i = 0; i < DIM; i++) {
        std::size_t j = 0;
        std::size_t b = i;
        std::size_t m = std::log2(DIM);
        while (m > 0) {
            j = j << 1;
            j = j + (b & 1);
            b = b >> 1;
            m--;
        }
        if (i < j) {
            scrambled[i] = j;
        } else
            scrambled[i] = i;
    }
    return scrambled;
}


template<typename floating_type, std::size_t DIM>
void bit_reverse_reorder(std::vector<std::complex<floating_type>> &W) {
    constexpr std::array<std::size_t, DIM> scrambled = bit_reverse_array<DIM>();
    for (std::size_t i = 0; i < W.size(); i++) {
        std::size_t j = scrambled[i];
        if (i < j) {
            swap(W[i], W[j]);
        }
    }
}


template<typename floating_type>
void recursiveDITFFT(std::vector<std::complex<floating_type>> &x) {
    using namespace std::complex_literals;
    auto x_size = x.size();

    if (x_size <= 1)
        return;

    std::vector<std::complex<floating_type>> odd;
    std::vector<std::complex<floating_type>> even;

    // decimation odd / even
    std::copy_if(x.cbegin(), x.cend(), std::back_inserter(odd),
                 [n = 0](std::complex<floating_type> c)mutable { return ((n++) % 2) != 0; });
    odd.shrink_to_fit();
    std::copy_if(x.cbegin(), x.cend(), std::back_inserter(even),
                 [n = 0](std::complex<floating_type> c)mutable { return ((n++) % 2) == 0; });
    even.shrink_to_fit();
    recursiveDITFFT(odd);
    recursiveDITFFT(even);

    for (std::size_t k = 0; k < x_size / 2; k++) {
        std::complex<floating_type> t = std::exp(-2. * 1i * std::numbers::pi * ((double) k / x_size)) * odd[k];
        x[k] = even[k] + t;
        x[x_size / 2 + k] = even[k] - t;
    }

}

template<typename floating_type, std::size_t DIM>
void iterativeDITFFT(std::vector<std::complex<floating_type>> &x,
                     const std::array<std::complex<floating_type>, DIM / 2> &tf) {
    std::size_t problemSize = x.size();
    std::size_t stages = std::log2(problemSize);

    bit_reverse_reorder<floating_type, DIM>(x);

    for (std::size_t stage = 0; stage <= stages; stage++) {
        std::size_t currentSize = 1 << stage;
        std::size_t step = stages - stage;
        std::size_t halfSize = currentSize / 2;
        for (std::size_t k = 0; k < problemSize; k = k + currentSize) {
            for (std::size_t j = 0; j < halfSize; j++) {
                auto u = x[k + j];
                auto v = x[k + j + halfSize] * tf[j * (1 << step)];
                x[k + j] = (u + v);
                x[k + j + halfSize] = (u - v);
            }
        }
    }

}

template<typename floating_type, std::size_t DIM>
void iterativeDITFFT(std::vector<std::complex<floating_type>> &x) {
    std::array<std::complex<floating_type>, DIM / 2> tf = twiddle_factors<floating_type, DIM>();
    iterativeDITFFT<floating_type, DIM>(x, tf);
}

template<typename floating_type>
void recursiveDIFFFT(std::vector<std::complex<floating_type>> &x) {
    using namespace std::complex_literals;
    std::size_t x_size = x.size();

    if (x_size <= 1)
        return;

    std::vector<std::complex<floating_type>> left(x_size / 2);
    std::vector<std::complex<floating_type>> right(x_size / 2);

    // decimation left / right
    for (std::size_t idx = 0; idx < x_size / 2; idx++) {
        left[idx] = x[idx];
        right[idx] = x[(x_size / 2) + idx];
    }

    for (std::size_t k = 0; k < x_size / 2; k++) {
        auto tmp = left[k];
        left[k] = left[k] + right[k];
        right[k] = (tmp - right[k]) * std::exp(-2. * 1i * std::numbers::pi * ((double) k / x_size));
    }

    recursiveDIFFFT(left);
    recursiveDIFFFT(right);

    std::size_t j = 0;
    for (std::size_t k = 0; k < x_size / 2; k++) {
        x[j] = left[k];
        x[j + 1] = right[k];
        j += 2;
    }

}

template<typename floating_type, std::size_t DIM>
void iterativeDIFFFT(std::vector<std::complex<floating_type>> &x,
                     const std::array<std::complex<floating_type>, DIM / 2> &tf) {
    std::size_t problemSize = x.size(), numOfProblem = 1;

    while (problemSize > 1) {
        std::size_t halfSize = problemSize / 2;
        for (std::size_t k = 0; k < numOfProblem; k++) {
            std::size_t jFirst = k * problemSize, jLast = jFirst + halfSize - 1, jTwiddle = 0;
            for (std::size_t j = jFirst; j <= jLast; j++) {
                auto W = tf[jTwiddle];
                auto temp = x[j];
                x[j] = temp + x[j + halfSize];
                x[j + halfSize] = W * (temp - x[j + halfSize]);
                jTwiddle += numOfProblem;
            }
        }
        numOfProblem *= 2;
        problemSize = halfSize;
    }
    bit_reverse_reorder<floating_type, DIM>(x);
}

template<typename floating_type, std::size_t DIM>
void iterativeDIFFFT(std::vector<std::complex<floating_type>> &x) {
    std::array<std::complex<floating_type>, DIM / 2> tf = twiddle_factors<floating_type, DIM>();
    iterativeDIFFFT<floating_type, DIM>(x, tf);
}

template<typename floating_type, std::size_t DIM>
constexpr std::array<floating_type, DIM> hamming_window() {
    std::array<floating_type, DIM> w{};
    std::generate(w.begin(), w.end(),
                  [index = 0]()mutable {
                      return (0.54 - 2 * 0.23 * std::cos(2 * std::numbers::pi * index++ / DIM));
                  });
    return w;
}

template<typename floating_type, std::size_t DIM>
void windowing(const std::array<floating_type, DIM> &w, std::vector<std::complex<floating_type>> &s) {
    std::transform(s.begin(),
                   s.end(),
                   s.begin(),
                   [&, index = 0](std::complex<floating_type> c)mutable {
                       return w[index++] * c;
                   });

}


#endif //AUFILEPROCESSING_FFT_H

#ifndef AUFILEPROCESSING_AUFILE_H
#define AUFILEPROCESSING_AUFILE_H

#define FFT_SIZE 512
#define FLOATING_TYPE double

#define KiB 1024
#define MiB 1024*1024
#define GiB 1024*1024*1024

#include <filesystem>
#include <vector>
#include <complex>
//#include "fourier_transforms/fft.h"


class AUFile {
public:
    explicit AUFile(const std::filesystem::path &filePath, bool bigEndianness = true, bool verbose = true);

    friend std::ostream &operator<<(std::ostream &s, const AUFile &h);

    friend std::ostream &operator<<(std::ostream &s, const AUFile *h);

    void export_csv(const std::string &csv_file) const;

    std::vector<FLOATING_TYPE> getFeatures() const;

    void featuresNormalize(const std::vector<FLOATING_TYPE> &avg, const std::vector<FLOATING_TYPE> &stddev);

private:
    static uint64_t read_word(std::ifstream &file, std::size_t byte_to_read = 4, bool bigEndian = true);

    void read_data(std::ifstream &file, bool bigEndian = true);

    void process_signal();

    const std::array<FLOATING_TYPE, FFT_SIZE> fft_window;
    const std::array<std::complex<FLOATING_TYPE>, FFT_SIZE / 2> tf;
    bool bigEndianness;
    bool verbose;

    std::vector<FLOATING_TYPE> audioData;

    std::vector<std::vector<std::complex<FLOATING_TYPE>>> bins;

    std::vector<FLOATING_TYPE> bins_average;
    std::vector<FLOATING_TYPE> bins_standard_deviation;
    std::vector<FLOATING_TYPE> features;

    std::string musicStyle;
    std::filesystem::path filePath;

    uint64_t magic_number;
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t encoding;
    uint64_t sample_rate;
    uint64_t channels;


};

#endif //AUFILEPROCESSING_AUFILE_H

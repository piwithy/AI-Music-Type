#include "AUFile.h"

#include <iostream>
#include <fstream>
#include <iterator>
#include <execution>
#include "fourier_transforms/fft.h"

AUFile::AUFile(const std::filesystem::path &filePath, bool bigEndianness, bool verbose) : fft_window(
        hamming_window<FLOATING_TYPE, FFT_SIZE>()), tf(twiddle_factors<FLOATING_TYPE, FFT_SIZE>()),
                                                                                          bigEndianness(bigEndianness),
                                                                                          verbose(verbose),
                                                                                          bins(),
                                                                                          bins_average(FFT_SIZE / 2, 0),
                                                                                          bins_standard_deviation(
                                                                                                  FFT_SIZE / 2, 0),
                                                                                          features(),
                                                                                          filePath(filePath) {
    if (verbose)
        std::cout << "Processing: \"" << filePath.string() << "\",";
    if (filePath.filename().extension() != ".au") {
        std::cerr << "\nError While Processing: " << filePath.filename() << std::endl;
        throw std::runtime_error("Not a .au FILE!");
    }

    //Detecting Music Style
    std::string f_path = filePath.filename().string();
    std::size_t pos = f_path.find('.');
    this->musicStyle = f_path.substr(0, pos);

    // ---------------------------------------------------------
    // | START OF READING TIMER                                |
    // ---------------------------------------------------------
    auto start_time = std::chrono::high_resolution_clock::now();

    // Opening Audio File
    std::ifstream audioFile(this->filePath);

    // Processing File Header
    this->magic_number = read_word(audioFile, 4, bigEndianness);
    if (magic_number != 0x2e736e64) {
        if (verbose)
            std::cerr << "Bad Endianness! (actual: Big Endian=" << std::boolalpha << bigEndianness << ")"
                      << std::endl;
        this->bigEndianness = !bigEndianness;
        audioFile.seekg(0, std::ios_base::beg);
        this->magic_number = read_word(audioFile, 4, this->bigEndianness);
        if (magic_number != 0x2e736e64)
            throw std::runtime_error("Bad File Encoding!");
        /*if (verbose)
            std::cout << "magic_number: " << std::hex << this->magic_number << std::dec << std::endl;*/
    }
    this->data_offset = read_word(audioFile, 4, this->bigEndianness);
    this->data_size = read_word(audioFile, 4, this->bigEndianness);
    this->encoding = read_word(audioFile, 4, this->bigEndianness);
    this->sample_rate = read_word(audioFile, 4, this->bigEndianness);
    this->channels = read_word(audioFile, 4, this->bigEndianness);
    read_data(audioFile);

    // ---------------------------------------------------------
    // | START OF PROCESSING TIMER                             |
    // ---------------------------------------------------------
    auto end_read = std::chrono::high_resolution_clock::now();
    process_signal();
    auto stop_time = std::chrono::high_resolution_clock::now();
    // ---------------------------------------------------------
    // | START OF PROCESSING TIMER                             |
    // ---------------------------------------------------------

    features.insert(features.begin(), bins_average.begin(), bins_average.end());
    features.insert(features.end(), bins_standard_deviation.begin(), bins_standard_deviation.end());

    if (audioFile.is_open())
        audioFile.close();
    auto duration_read_process = (stop_time - start_time) / std::chrono::milliseconds(1);
    auto duration_read = (end_read - start_time) / std::chrono::milliseconds(1);
    auto duration_process = (stop_time - end_read) / std::chrono::milliseconds(1);
    if (verbose)
        std::cout << " file read: " << duration_read << "ms, " << "Audio process: "
                  << duration_process << "ms, "
                  << "Total: " << duration_read_process << "ms" << std::endl;

}

uint64_t AUFile::read_word(std::ifstream &file, std::size_t byte_to_read, bool bigEndian) {
    uint64_t word = 0;
    uint8_t b;
    if (byte_to_read > 8)
        throw std::invalid_argument("to much bytes to read (8 max)");
    for (std::size_t i = 0; i < byte_to_read; i++) {
        file.read(reinterpret_cast<char *>(&b), sizeof(uint8_t));
        if (bigEndian)
            word = word | (b << (((byte_to_read - 1) - i) * 8));
        else
            word = word | (b << (i * 8));
    }
    return word;

}

void AUFile::read_data(std::ifstream &file, bool bigEndian) {
    std::size_t word_size = 0;
    if (encoding == 3)
        word_size = 2;
    file.seekg(data_offset, std::ios_base::beg);
    for (size_t data_idx = 0; data_idx < data_size / word_size; data_idx++) {
        audioData.push_back((float) ((short) read_word(file, word_size, bigEndian)));
    }
    audioData.shrink_to_fit();
}

void AUFile::process_signal() {
    std::size_t n_chunks = (audioData.size() / FFT_SIZE);
    std::vector<std::complex<FLOATING_TYPE>> cplx_data(audioData.size(), std::complex<FLOATING_TYPE>(0., 0.));
    std::transform(std::execution::par, audioData.cbegin(), audioData.cend(), cplx_data.begin(),
                   [](FLOATING_TYPE value) {
                       return std::complex<FLOATING_TYPE>(value, 0);
                   });
    for (std::size_t k = 0; k < cplx_data.size() / FFT_SIZE; k++) {
        //std::cout << "k=" << k << std::endl;
        // Preparing working vectors
        std::vector<std::complex<FLOATING_TYPE>> v1(FFT_SIZE, std::complex<FLOATING_TYPE>(0.0, 0.0));
        std::vector<std::complex<FLOATING_TYPE>> v2(FFT_SIZE, std::complex<FLOATING_TYPE>(0.0, 0.0));
        std::copy(cplx_data.cbegin() + k * FFT_SIZE, cplx_data.cbegin() + FFT_SIZE * (k + 1), v1.begin());
        std::copy(cplx_data.cbegin() + FFT_SIZE * (k + 0.5), cplx_data.cbegin() + FFT_SIZE * (k + 1.5), v2.begin());

        //windowing
        windowing(fft_window, v1);
        windowing(fft_window, v2);

        //Computing FFT
        iterativeDITFFT<FLOATING_TYPE, FFT_SIZE>(v1, tf);
        iterativeDITFFT<FLOATING_TYPE, FFT_SIZE>(v2, tf);

        // Adding to bins
        bins.push_back(v1);
        bins.push_back(v2);
    }

    std::transform(bins_standard_deviation.cbegin(), bins_standard_deviation.cend(), bins_standard_deviation.begin(),
                   [n_chunks](auto bin_val) { return std::sqrt(bin_val / (double) n_chunks); });


    bins.shrink_to_fit();

    for (std::size_t frq_idx = 0; frq_idx < FFT_SIZE / 2; frq_idx++) {
        std::vector<double> bin_value(bins.size(), 0.);
        for (std::size_t bin_idx = 0; bin_idx < bins.size(); bin_idx++) {
            bin_value[bin_idx] = std::abs(bins[bin_idx][frq_idx]);
        }
        double avg = std::reduce(std::execution::par, bin_value.cbegin(), bin_value.cend(), 0.) /
                     (double) bin_value.size();
        bins_average[frq_idx] = avg;
        bins_standard_deviation[frq_idx] = std::sqrt(std::transform_reduce(std::execution::par,
                                                                           bin_value.cbegin(),
                                                                           bin_value.cend(),
                                                                           0.,
                                                                           std::plus(),
                                                                           [avg](double x) {
                                                                               return ((x - avg) * (x - avg));
                                                                           }) / (double) (bin_value.size() - 1));
    }
}


void AUFile::export_csv(const std::string &csv_file) const {
    std::ofstream outFile(csv_file, std::ios::app);
    const std::vector<FLOATING_TYPE> &sound_features = this->features;

    if (!sound_features.empty()) {
        std::copy(sound_features.cbegin(), sound_features.cend() - 1,
                  std::ostream_iterator<FLOATING_TYPE>(outFile, ","));
        outFile << sound_features.back() << ",";
    }
    outFile << "\"" << this->musicStyle << "\"\n";
}

std::ostream &operator<<(std::ostream &s, const AUFile &h) {
    char magic_str[5] = {
            (char) ((h.magic_number & 0xFF000000) >> 24u),
            (char) ((h.magic_number & 0x00FF0000) >> 16u),
            (char) ((h.magic_number & 0x0000FF00) >> 8u),
            (char) ((h.magic_number & 0x000000FF) >> 0u),
            '\0'
    };
    float data_size_conv;
    std::string size_unit;
    if (h.data_size < KiB) {
        data_size_conv = h.data_size;
        size_unit = "B";
    } else if (h.data_size < MiB) {
        data_size_conv = (float) h.data_size / KiB;
        size_unit = "KiB";
    } else if (h.data_size < GiB) {
        data_size_conv = (float) h.data_size / MiB;
        size_unit = "MiB";
    } else {
        data_size_conv = (float) h.data_size / GiB;
        size_unit = "GiB";
    }
    return s << "\"" << h.filePath.string() << "\" Header:\n"
             << std::hex << "\tMagic Number\t= 0x" << h.magic_number << "\n"
             << "\tMagic String\t= \"" << magic_str << "\"\n"
             << std::dec << "\tData Offset\t\t= " << h.data_offset << "\n"
             << std::dec << "\tData size\t\t= " << data_size_conv << size_unit << "\n"
             << std::dec << "\tEncoding\t\t= " << h.encoding << "\n"
             << std::dec << "\tSample Rate\t\t= " << h.sample_rate << " sample/s \n"
             << std::dec << "\tChannels\t\t= " << h.channels
             << std::dec;
}

std::ostream &operator<<(std::ostream &s, const AUFile *h) {
    char magic_str[5] = {
            (char) ((h->magic_number & 0xFF000000) >> 24u),
            (char) ((h->magic_number & 0x00FF0000) >> 16u),
            (char) ((h->magic_number & 0x0000FF00) >> 8u),
            (char) ((h->magic_number & 0x000000FF) >> 0u),
            '\0'
    };
    float data_size_conv;
    std::string size_unit;
    if (h->data_size < KiB) {
        data_size_conv = h->data_size;
        size_unit = " B";
    } else if (h->data_size < MiB) {
        data_size_conv = (float) h->data_size / KiB;
        size_unit = " KiB";
    } else if (h->data_size < GiB) {
        data_size_conv = (float) h->data_size / MiB;
        size_unit = " MiB";
    } else {
        data_size_conv = (float) h->data_size / GiB;
        size_unit = " GiB";
    }
    return s << "\"" << h->filePath.string() << "\" Header:\n"
             << std::hex << "\tMagic Number\t= 0x" << h->magic_number << "\n"
             << "\tMagic String\t= \"" << magic_str << "\"\n"
             << std::dec << "\tData Offset\t\t= " << h->data_offset << "\n"
             << std::dec << "\tData size\t\t= " << data_size_conv << size_unit << "\n"
             << std::dec << "\tEncoding\t\t= " << h->encoding << "\n"
             << std::dec << "\tSample Rate\t\t= " << h->sample_rate << " sample/s \n"
             << std::dec << "\tChannels\t\t= " << h->channels
             << std::dec;
}

std::vector<double> AUFile::getFeatures() const {
    return features;
}

void AUFile::featuresNormalize(const std::vector<FLOATING_TYPE> &avg, const std::vector<FLOATING_TYPE> &stddev) {
    for (std::size_t featuresInc = 0; features.begin() + featuresInc != features.end(); featuresInc++) {
        features[featuresInc] = (features[featuresInc] - avg[featuresInc]) / stddev[featuresInc];
    }
}



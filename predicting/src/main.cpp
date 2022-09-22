//
// Created by salyd on 20/02/2021.
//
#include <iostream>
#include <nlohmann/json.hpp>
#include <AUFile.h>
#include <iterator>
#include <vector>
#include <getopt.h>
#include <sstream>
#include <chrono>
#include <fstream>

using json = nlohmann::json;

template<typename T>
std::ostream &operator<<(std::ostream &s, const std::vector<T> &v) {
    s << '[';
    if (!v.empty()) {
        std::copy(v.cbegin(), v.cend() - 1, std::ostream_iterator<T>(s, ", "));
        s << v.back();
    }
    return s << ']';
}

template<typename T_first, typename T_second>
std::ostream &operator<<(std::ostream &s, const std::pair<T_first, T_second> &p) {
    return s << "<" << p.first << ", " << p.second << ">";
}


int main(int argc, char *argv[]) {
    std::stringstream ss;
    ss << "Usage: " << argv[0] << " [-v] -m <model_file> -a <audio_file>";
    std::string usage_message = ss.str();
    bool verbose = false;
    if (argc == 1) {
        std::cerr << usage_message << std::endl;
        exit(EXIT_FAILURE);
    }
    int opt;
    std::string model_path, audio_file_path;
    while ((opt = getopt(argc, argv, "m:a:hv")) != -1) {
        switch (opt) {
            case 'm':
                model_path = std::string(optarg);
                break;
            case 'a':
                audio_file_path = std::string(optarg);
                break;
            case 'v':
                verbose = true;
                break;
            case 'h':
                std::cout << usage_message << std::endl;
                exit(EXIT_SUCCESS);
            default:
                std::cerr << usage_message << std::endl;
                exit(EXIT_FAILURE);
        }
    }
    if (model_path.empty() || audio_file_path.empty()) {
        std::cerr << usage_message << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Predicting W/ Model from: \"" << model_path << "\"" << std::endl;
    std::cout << "Predicting style of: \"" << audio_file_path << "\"" << std::endl;
    if (verbose)
        std::cout << "Starting model import!" << std::endl;
    auto start_model_import = std::chrono::high_resolution_clock::now();
    std::vector<double> intercepts;
    std::vector<std::vector<double>> coeffs;
    std::vector<double> std_norm, avg_norm;
    json model_json;
    std::ifstream model_file(model_path);
    model_file >> model_json;
    for (const auto &inter: model_json["intercept"]) {
        intercepts.push_back((double) inter);
    }
    for (const auto &class_coef : model_json["coef"]) {
        std::vector<double> local_coeff;
        local_coeff.clear();
        for (const auto &coef : class_coef) {
            local_coeff.push_back((double) coef);
        }
        coeffs.push_back(local_coeff);
    }
    for (const auto &avg: model_json["normalisation"]["average"]) {
        avg_norm.push_back((double) avg);
    }
    for (const auto &stddev: model_json["normalisation"]["std"]) {
        std_norm.push_back((double) stddev);
    }
    auto finish_model_import = std::chrono::high_resolution_clock::now();
    auto duration_import_model = (finish_model_import - start_model_import) / std::chrono::milliseconds(1);
    if (verbose)
        std::cout << "Model imported in " << duration_import_model << "ms" << std::endl;
    std::cout << "Start the processing of the file to guess the style of" << std::endl;
    auto start_audio_processing = std::chrono::high_resolution_clock::now();
    AUFile auFile(std::filesystem::path(audio_file_path), true, verbose);
    auFile.featuresNormalize(avg_norm, std_norm);
    auto finish_audio_processing = std::chrono::high_resolution_clock::now();
    auto duration_audio_process = (finish_audio_processing - start_audio_processing) / std::chrono::milliseconds(1);
    if (verbose)
        std::cout << "Audio processed in " << duration_audio_process << "ms" << std::endl;
    auto start_prediction = std::chrono::high_resolution_clock::now();
    std::vector<std::pair<int, double>> class_ranking;
    auto audioFeatures = auFile.getFeatures();
    std::pair<int, double> max_class = std::make_pair(-1, -5000.);
    for (std::size_t class_idx = 0; class_idx < coeffs.size(); class_idx++) {
        auto class_coef = coeffs[class_idx];
        std::vector<double> tmp_proba;
        double proba =
                std::transform_reduce(class_coef.cbegin(), class_coef.cend(), audioFeatures.cbegin(), 0.0, std::plus(),
                                      [](double a, double b) { return a * b; }) - intercepts[class_idx];
        if (max_class.second == -5000.)
            max_class = std::make_pair(class_idx, proba);
        if (proba > max_class.second) {
            max_class = std::make_pair(class_idx, proba);
        }
        class_ranking.emplace_back(class_idx, proba);
        if (verbose)
            std::cout << "Factor: " << (double) round(proba * 1000) / 1000 << "\t\tClass: "
                      << model_json["classes"][std::to_string(class_idx)] << std::endl;
    }
    auto end_prediction = std::chrono::high_resolution_clock::now();
    auto duration_prediction = (end_prediction - start_prediction) / std::chrono::microseconds(1);

    if (verbose)
        std::cout << "Predicted in: " << duration_prediction << "µs" << std::endl;

    std::cout << "Best guess on music style for " << audio_file_path << " is "
              << model_json["classes"][std::to_string(max_class.first)] << " W/ a Factor of: "
              << round(max_class.second * 1000) / (double) 1000
              << std::endl;
    return EXIT_SUCCESS;
}

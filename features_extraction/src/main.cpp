//
// Created by salyd on 20/02/2021.
//
#include <iostream>
#include <fstream>
#include <filesystem>
#include <getopt.h>
#include <AUFile.h>

namespace fs = std::filesystem;


void readFile(const fs::path &filePath, const std::string &exportFile, int &file_count, bool verbose) {
    if (fs::exists(filePath) && fs::is_regular_file(filePath)) {
        AUFile audioFile(filePath, true, verbose);
        audioFile.export_csv(exportFile);
        file_count++;
    }
}

void readDir(const fs::path &filePath, const std::string &export_file, int &file_count, bool verbose) {
    if (fs::exists(filePath) && fs::is_directory(filePath)) {
        for (const auto &entry: fs::directory_iterator(filePath)) {
            if (fs::is_directory(entry)) {
                readDir(entry, export_file, file_count, verbose);
            } else {
                readFile(entry, export_file, file_count, verbose);
            }
        }
    } else {
        std::cerr << filePath << "Not Valid" << std::endl;
        throw std::runtime_error("Invalid File Path");
    }
}

double round(double to_round, int digits = 0) {
    return (int) (to_round * pow(10, digits)) / (double) (pow(10, digits));
}

int main(int argc, char *argv[]) {
    std::stringstream ss;
    ss << "Usage: " << argv[0] << " [-v] -i <assets_folder> -o <features_csv>";
    std::string usage_message = ss.str();
    bool verbose = false;
    int file_count = 0;
    /*std::cout << "AU Audio File Feature Extractor v" << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH
              << std::endl;*/
    if (argc == 1) {
        std::cerr << usage_message << std::endl;
        exit(EXIT_FAILURE);
    }
    int opt;
    std::string input_path, output_path;
    while ((opt = getopt(argc, argv, "i:o:hv")) != -1) {
        switch (opt) {
            case 'i':
                input_path = std::string(optarg);
                break;
            case 'o':
                output_path = std::string(optarg);
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
    if (input_path.empty() || output_path.empty()) {
        std::cerr << usage_message << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Extraction Features from: \"" << input_path << "\"" << std::endl;
    std::cout << "Exporting  Features to  : \"" << output_path << "\"" << std::endl;
    std::cout << "Starting extraction" << std::endl;
    fs::path audioFilesPath(input_path);
    auto start_time = std::chrono::high_resolution_clock::now();
    std::ofstream outfile(output_path);
    outfile << "";
    outfile.close();
    readDir(audioFilesPath, output_path, file_count, verbose);
    auto end_time = std::chrono::high_resolution_clock::now();
    //auto duration_us = (end_time - start_time) / std::chrono::microseconds(1);
    auto duration_ms = (end_time - start_time) / std::chrono::milliseconds(1);
    //auto duration_s = (end_time - start_time) / std::chrono::seconds(1);
    std::cout << "Analysed " << file_count << " files in " << round((double) duration_ms / 1000, 3) << "s ("
              << (double) duration_ms / (double) file_count << " ms per file)" << std::endl;
    return EXIT_SUCCESS;
}



#include "TestMatrixBasedModelSelector.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>

TestMatrixBasedModelSelector::TestMatrixBasedModelSelector(int n, int d, double delta, int B, int H)
    : n(n), d(d), delta(delta), B(B), H(H) {}

std::vector<std::vector<int>> TestMatrixBasedModelSelector::generate_rrsd_matrix() {
    // (Same as before...)
}

std::set<int> TestMatrixBasedModelSelector::decode(const std::vector<std::vector<int>>& matrix, const std::set<int>& defective_set) {
    // (Same as before...)
}

std::pair<std::vector<std::vector<int>>, double> TestMatrixBasedModelSelector::find_best_model() {
    std::vector<std::vector<int>> best_matrix;
    double best_score = -1.0;

    for (int b = 0; b < B; ++b) {
        auto matrix = generate_rrsd_matrix();
        double total_score = 0.0;
        
        for (int h = 0; h < H; ++h) {
            std::set<int> defective_set;
            while (defective_set.size() < d) {
                defective_set.insert(rand() % n);
            }
            auto decoded_set = decode(matrix, defective_set);

            std::set<int> difference;
            std::set_difference(decoded_set.begin(), decoded_set.end(),
                                defective_set.begin(), defective_set.end(),
                                std::inserter(difference, difference.begin()));

            int false_positives = difference.size();
            total_score += (1.0 - static_cast<double>(false_positives) / n);
        }
        
        double average_score = total_score / H;
        if (average_score > best_score) {
            best_score = average_score;
            best_matrix = matrix;
        }
    }

    return {best_matrix, best_score};
}

int main() {
    TestMatrixBasedModelSelector selector(10, 2, 0.1, 10, 10);
    auto [best_matrix, best_score] = selector.find_best_model();

    std::ofstream matrix_file("best_rrsd_matrix.txt");
    for (const auto& row : best_matrix) {
        for (int val : row) {
            matrix_file << val << " ";
        }
        matrix_file << "\n";
    }
    matrix_file.close();

    std::ofstream score_file("best_score.txt");
    score_file << best_score;
    score_file.close();

    return 0;
}

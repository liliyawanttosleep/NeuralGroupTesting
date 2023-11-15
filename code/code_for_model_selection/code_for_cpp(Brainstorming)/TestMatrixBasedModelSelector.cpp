
#include "TestMatrixBasedModelSelector.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>

TestMatrixBasedModelSelector::TestMatrixBasedModelSelector(int n, int d, double delta, int B, int H)
    : n(n), d(d), delta(delta), B(B), H(H) {}

std::vector<std::vector<int>> TestMatrixBasedModelSelector::generate_rrsd_matrix() {
    double p = std::exp(-1.0 / d);
    int r = static_cast<int>((1 - p) * (n - d + 1));
    int m = static_cast<int>(std::exp(1) * d * std::log(2 * n / delta));
    std::vector<std::vector<int>> matrix(m, std::vector<int>(n, 0));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);

    for (int i = 0; i < m; ++i) {
        std::set<int> random_columns;
        while (random_columns.size() < r) {
            random_columns.insert(dis(gen));
        }
        for (int col : random_columns) {
            matrix[i][col] = 1;
        }
    }
    return matrix;
}

std::set<int> TestMatrixBasedModelSelector::decode(const std::vector<std::vector<int>>& matrix, const std::set<int>& defective_set) {
    std::set<int> X;
    for (int i = 0; i < matrix[0].size(); ++i) {
        X.insert(i);
    }

    for (const auto& row : matrix) {
        bool all_zero = true;
        for (int i : defective_set) {
            if (row[i] != 0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero) {
            for (int i = 0; i < row.size(); ++i) {
                if (row[i] == 1) {
                    X.erase(i);
                }
            }
        }
    }
    return X;
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
            int false_positives = std::distance(std::set_difference(decoded_set.begin(), decoded_set.end(),
                                                         defective_set.begin(), defective_set.end(),
                                                         std::inserter(std::set<int>(), std::set<int>().begin())), decoded_set.end());
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

    // Output the best matrix and score for demonstration
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

    // Output the best matrix and score for demonstration
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

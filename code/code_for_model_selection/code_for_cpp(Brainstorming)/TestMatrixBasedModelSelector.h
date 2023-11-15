
#ifndef TEST_MATRIX_BASED_MODEL_SELECTOR_H
#define TEST_MATRIX_BASED_MODEL_SELECTOR_H

#include <vector>
#include <set>

class TestMatrixBasedModelSelector {
public:
    TestMatrixBasedModelSelector(int n, int d, double delta, int B=100, int H=100);
    std::vector<std::vector<int>> generate_rrsd_matrix();
    std::set<int> decode(const std::vector<std::vector<int>>& matrix, const std::set<int>& defective_set);
    std::pair<std::vector<std::vector<int>>, double> find_best_model();

private:
    int n, d, B, H;
    double delta;
};

#endif // TEST_MATRIX_BASED_MODEL_SELECTOR_H

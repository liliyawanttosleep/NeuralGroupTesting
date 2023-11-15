
import numpy as np
import random

# Function to generate a matrix based on parameters n, d, and delta
def generate_rrsd_matrix(n, d, delta):
    p = np.exp(-1/d)
    r = int((1-p)*(n-d+1))
    m = int(np.exp(1)*d*np.log(2*n/delta))
    matrix = np.zeros((m, n))
    for i in range(m):
        random_columns = np.random.choice(n, r, replace=False)
        matrix[i, random_columns] = 1
    return matrix

# Function to generate H random sets of size d
def generate_defective_sets(n, d, H):
    defective_sets = []
    for _ in range(H):
        defective_set = random.sample(range(n), d)
        defective_sets.append(defective_set)
    return defective_sets

# Function to calculate the number of false positives for a given defective set and matrix
def calculate_false_positives(matrix, defective_set, n):
    X = set(range(n))
    for row in matrix:
        test_result = sum(row[i] for i in defective_set)
        if test_result == 0:
            X -= set(i for i, val in enumerate(row) if val == 1)
    false_positives = len(X - set(defective_set))
    return false_positives

# Function to calculate score for a single defective set
def calculate_score(n, false_positives):
    return (1 - false_positives/n)

# Main function to execute the entire flow
def find_best_model(B=100, H=10, n_range=(10, 100), d_range=(1, 10), delta_range=(0.1, 1)):
    best_model = None
    best_score = -1
    for _ in range(B):
        n = random.randint(*n_range)
        d = random.randint(*d_range)
        delta = random.uniform(*delta_range)
        matrix = generate_rrsd_matrix(n, d, delta)
        defective_sets = generate_defective_sets(n, d, H)
        total_score = 0
        for defective_set in defective_sets:
            false_positives = calculate_false_positives(matrix, defective_set, n)
            total_score += calculate_score(n, false_positives)
        average_score = total_score / H
        if average_score > best_score:
            best_score = average_score
            best_model = {
                'n': n,
                'd': d,
                'delta': delta,
                'matrix': matrix
            }
    return best_model

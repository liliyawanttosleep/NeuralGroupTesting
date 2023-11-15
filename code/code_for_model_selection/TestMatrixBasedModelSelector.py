
import numpy as np
import random

class TestMatrixBasedModelSelector:
    def __init__(self, n, d, delta, B=100, H=100):
        self.n = n
        self.d = d
        self.delta = delta
        self.B = B
        self.H = H

    def generate_rrsd_matrix(self):
        p = np.exp(-1 / self.d)
        r = int((1 - p) * (self.n - self.d + 1))
        m = int(np.exp(1) * self.d * np.log(2 * self.n / self.delta))
        matrix = np.zeros((m, self.n))
        for i in range(m):
            random_columns = np.random.choice(self.n, r, replace=False)
            matrix[i, random_columns] = 1
        return matrix

    def generate_defective_sets(self):
        defective_sets = []
        for _ in range(self.H):
            defective_set = random.sample(range(self.n), self.d)
            defective_sets.append(defective_set)
        return defective_sets

    def decode(self, matrix, defective_set):
        X = set(range(self.n))
        for row in matrix:
            if all(row[i] == 0 for i in defective_set):
                X -= {i for i, val in enumerate(row) if val == 1}
        return X

    def calculate_false_positives(self, matrix, defective_set):
        X = self.decode(matrix, defective_set)
        false_positives = len(X - set(defective_set))
        return false_positives

    def calculate_score(self, false_positives):
        return (1 - false_positives / self.n)

    def find_best_model(self):
        best_matrix = None
        best_score = -1
        defective_sets = self.generate_defective_sets()
        
        for b in range(self.B):
            matrix = self.generate_rrsd_matrix()
            total_score = 0
            correct_decodes = 0
            print(f"For Test Matrix {b + 1}:")
            
            for defective_set in defective_sets:
                false_positives = self.calculate_false_positives(matrix, defective_set)
                decoded_set = self.decode(matrix, defective_set)
                
                if set(defective_set) == decoded_set:
                    correct_decodes += 1

                #print(f"Defective Set: {defective_set}")
                #print(f"Decoded Set: {decoded_set}")
                
                total_score += self.calculate_score(false_positives)
            
            average_score = total_score / self.H
            print(f"Correctly Decoded: {correct_decodes} out of {self.H}")
            print(f"Current score:{average_score}")
            
            if average_score > best_score:
                best_score = average_score
                best_matrix = matrix
                
        return {'best_matrix': best_matrix, 'best_score': best_score}

# Example usage
if __name__ == "__main__":
    selector = TestMatrixBasedModelSelector(n=2048, d=16, delta=0.1, B=100, H=100)
    result = selector.find_best_model()
    best_matrix = result['best_matrix']
    best_score = result['best_score']
    print("Best Matrix:")
    print(best_matrix)
    print("Best Score:", best_score)
    print("Best matrix shape {}".format(best_matrix.shape))

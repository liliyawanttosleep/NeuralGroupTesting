
import numpy as np
import random

class BestModelFinderWithTesting(BestModelFinder):
    def generate_rrsd_matrix(self):
        """
        Generate an RRSD matrix based on the initialized parameters n, d, and delta.
        """
        p = np.exp(-1 / self.d)
        r = int((1 - p) * (self.n - self.d + 1))
        m = int(np.exp(1) * self.d * np.log(2 * self.n / self.delta))
        print(f"Generating RRSD matrix with m={m}, n={self.n}, r={r}")
        matrix = np.zeros((m, self.n))
        for i in range(m):
            random_columns = np.random.choice(self.n, r, replace=False)
            matrix[i, random_columns] = 1
        return matrix

    def decode(self, matrix, defective_set):
        """
        Decode a given defective set using the provided matrix.
        """
        print(f"Decoding defective set: {defective_set}")
        X = set(range(self.n))
        for row in matrix:
            print(f"Processing row: {row}")
            if all(row[i] == 0 for i in defective_set):
                print("Row gives a negative result for the defective set.")
                X -= {i for i, val in enumerate(row) if val == 1}
        print(f"Decoded set X: {X}")
        return X

import numpy as np

def generate_matrix(n, m):
    # Step 1: Generate a vector L of length n with corresponding probabilities
    L = np.random.randint(1, 10000, n)  # Random integers for x_i
    p = np.random.uniform(0.0001, 0.9999, n)  # Random probabilities for p_i
    
    # Initialize the matrix with zeros
    matrix = np.zeros((m, n), dtype=int)
    
    # Initialize the set M to store vectors y_i
    M = []
    
    # Step 2 and 4: Generate the binary matrix and collect y_i vectors
    for i in range(m):
        # Randomly pick 8 indices for setting "1"
        rand_indices = np.random.choice(n, 16, replace=False)
        matrix[i, rand_indices] = 1
        
        # Calculate the average probability of selected 8 indices
        avg_prob = np.mean(p[rand_indices])
        
        # If the average probability is greater than 0.5, add to set M
        if avg_prob > 0.6:
            M.append(L[rand_indices])
    
    # Step 3: Adding x_i and p_i as metadata (for illustration, not part of the actual matrix)
    metadata = np.vstack((L, p))
    
    return matrix, metadata, M

# Test the function
n = 10000  # n should be greater than 40
m = 1000  # Number of rows for the initial matrix

matrix, metadata, M = generate_matrix(n, m)

# Display results
print("Generated Matrix:")
print(matrix)

print("\nMetadata (First row is x_i, Second row is p_i):")
print(metadata)

print("\nSet M containing vectors y_i:")
print(M)
# Re-running the function to generate the initial matrix and set M




# Define the function to generate the matrix M1 and set M1

def generate_matrix_M1(M, p, L, m):
    # Flatten the vectors in set M
    flattened_M = np.array(M).flatten()
    unique_elements = np.unique(flattened_M)
    
    # Map unique elements of flattened M to their probabilities from p
    p_new = np.array([p[np.where(L == elem)[0][0]] for elem in unique_elements])

    # Define the new matrix dimensions
    n_1 = len(unique_elements)
    
    # Ensure m is greater than 20
    if m <= 20:
        return "m should be greater than 20"
    
    # Initialize an m x n_1 matrix with zeros
    matrix_M1 = np.zeros((m, n_1), dtype=int)
    
    # Initialize set M1 to store vectors
    M1 = []
    
    # Generate the binary matrix with each row of weight 4
    for i in range(m):
        # Randomly pick 4 unique indices to set to "1"
        rand_indices = np.random.choice(n_1, 8, replace=False)
        matrix_M1[i, rand_indices] = 1
        
        # Calculate average probability for the selected indices
        avg_prob = np.mean(p_new[rand_indices])
        
        # If the average probability is greater than 0.6, add this vector to set M1
        if avg_prob > 0.7:
            M1.append(unique_elements[rand_indices])
            
    return matrix_M1, M1

# Test the function with m > 20
m = 500  # m should be greater than 20

# Generate the matrix M1 and set M1
matrix_M1, M1 = generate_matrix_M1(M, metadata[1], metadata[0], m)

# Display the results
print("Generated Matrix M1:")
print(matrix_M1)

print("\nSet M1 containing vectors:")
for vec in M1:
    print(vec)
# Define the function to generate the matrix M2 and set M2

def generate_matrix_M2(M1, p, L, m):
    # Flatten the vectors in set M1
    flattened_M1 = np.array(M1).flatten()
    unique_elements = np.unique(flattened_M1)
    
    # Map unique elements of flattened M1 to their probabilities from p
    p_new = np.array([p[np.where(L == elem)[0][0]] for elem in unique_elements])

    # Define the new matrix dimensions
    n_2 = len(unique_elements)
    
    # Ensure m is greater than 20
    if m <= 20:
        return "m should be greater than 20"
    
    # Initialize an m x n_2 matrix with zeros
    matrix_M2 = np.zeros((m, n_2), dtype=int)
    
    # Initialize set M2 to store vectors
    M2 = []
    
    # Generate the binary matrix with each row of weight 4
    for i in range(m):
        # Randomly pick 4 unique indices to set to "1"
        rand_indices = np.random.choice(n_2, 4, replace=False)
        matrix_M2[i, rand_indices] = 1
        
        # Calculate average probability for the selected indices
        avg_prob = np.mean(p_new[rand_indices])
        
        # If the average probability is greater than 0.8, add this vector to set M2
        if avg_prob > 0.9:
            M2.append(unique_elements[rand_indices])
            
    return matrix_M2, M2

# Test the function with m > 20
m = 100  # m should be greater than 20

# Generate the matrix M2 and set M2
matrix_M2, M2 = generate_matrix_M2(M1, metadata[1], metadata[0], m)

# Display the results
print("Generated Matrix M2:")
print(matrix_M2)

print("\nSet M2 containing vectors:")
for vec in M2:
    print(vec)


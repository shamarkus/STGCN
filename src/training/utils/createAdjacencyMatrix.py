import numpy as np

def create_adjacency_matrix(joint_dimension):
	connectivity= [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0), (3, 6), (6, 7), (7, 8), (8, 9),  (3, 10), (10, 11), (11, 12), (12, 13),  (0, 14), (14, 15), (15, 16), (16, 17), (0, 18), (18, 19), (19, 20), (20, 21)]

	A = np.zeros((joint_dimension, joint_dimension)) # adjacency matrix
	for i, j in connectivity:
		if i < joint_dimension and j < joint_dimension:  
			A[j, i] = 1
			A[i, j] = 1
	return A

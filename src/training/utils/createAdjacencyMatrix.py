import numpy as np

def create_adjacency_matrix(joint_dimension):
	if joint_dimension == 22:
		connectivity = [(5, 4), (4, 3), (3, 2), (2, 1), (1, 0), (3, 6), (6, 7), (7, 8), (8, 9),  (3, 10), (10, 11), (11, 12), (12, 13),  (0, 14), (14, 15), (15, 16), (16, 17), (0, 18), (18, 19), (19, 20), (20, 21)]
	elif joint_dimension == 25:
		connectivity = [(0, 1), (0, 12) , (0, 16) , (1, 20) , (12, 13) , (13, 14) , (14, 15) , (16, 17) , (17, 18) , (18, 19) , (20, 4) , (20, 8) , (20, 2) , (2, 3) , (4, 5) , (5, 6) , (6, 7) , (6, 22) , (7, 21) , (8, 9) , (9, 10) , (10, 11) , (10, 24) , (11, 23)] 
		
	A = np.zeros((joint_dimension, joint_dimension)) # adjacency matrix
	for i, j in connectivity:
		if i < joint_dimension and j < joint_dimension:  
			A[j, i] = 1
			A[i, j] = 1
	return A

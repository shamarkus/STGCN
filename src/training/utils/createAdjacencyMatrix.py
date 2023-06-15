import numpy as np

def create_adjacency_matrix(joint_dimension):
  neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                    (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                    (22, 23), (23, 8), (24, 25), (25, 12)]
  
  neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
  edge = neighbor_link
  A = np.zeros((joint_dimension, joint_dimension)) # adjacency matrix
  for i, j in edge:
      if i < joint_dimension and j < joint_dimension:  
          A[j, i] = 1
          A[i, j] = 1
  return A

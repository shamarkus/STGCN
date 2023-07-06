import numpy as np
import pandas as pd
import pickle
import zipfile
import os
from scipy.spatial.transform import Rotation as R

def construct_skeleton_kinect(position, angle):
	'''
	Construct human skeleton from Kinect data.
	Position and angle are given relatively to each joint's father joint, except the root joint. 
	'''
	HUMAN_TREE = { 0: [1, 14, 18], 1: [2], 14: [15], 18: [19], 2: [3], 15: [16], 19: [20], 3: [4, 6, 10], 16: [17], 20: [21], 4: [5], 6: [7], 10: [11], 7: [8], 11: [12], 8: [9], 12: [13] }

	HUMAN_TREE_LAYER = [[0], [1, 14, 18], [2, 15, 19], [3, 16, 20], [4, 6, 10], [7, 11], [8, 12]]

	abs_position = np.zeros((position.shape[0], 3))
	abs_position[0] = position[0]
	abs_angle = np.zeros((angle.shape[0], 3, 3))

	abs_angle[0] = np.array( R.from_euler('xyz', angle[0], degrees=True).as_matrix())
	for layer in HUMAN_TREE_LAYER:
		for b in layer:
			for e in HUMAN_TREE[b]:
				abs_angle[e] = np.array( R.from_euler('xyz', angle[e], degrees=True).as_matrix())
				abs_angle[e] = np.matmul(abs_angle[e], abs_angle[b])

	for layer in HUMAN_TREE_LAYER:
		for b in layer:
			for e in HUMAN_TREE[b]:
				abs_position[e] = np.matmul(abs_angle[b], position[e]) + abs_position[b]

	rotmat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
	abs_position = np.matmul(abs_position, rotmat)
	return abs_position

def __augment__(sample, aug_angle):
	euler = (2 * np.random.rand(3) - 1) * (aug_angle / 180.) * np.pi
	rotation = R.from_euler('zxy', euler, degrees=False).as_matrix()
	sample = np.einsum('ij,kmj->kmi', rotation, sample)
	return sample

# Assuming a 1, 2, 5, 10 split
def getAugList(sample):
	result = []
	result.append(sample)

	for angle in [1, 2, 5, 10]:
		result.append(__augment__(sample,angle))
	return result

def processPRMD():

	np.random.seed(3770)

	mainPath = '../../../data/'
	rawPath = mainPath + 'raw/'
	cleanPath = mainPath + 'clean/'

	zip_files = [
		rawPath + 'corprmd.zip',
		rawPath + 'incprmd.zip'
	]

# 	for zip_file in zip_files:
# 		with zipfile.ZipFile(zip_file, 'r') as zip_ref:
# 			zip_ref.extractall(rawPath)

	prmdDataset = []

  # Correct Segmented Movements
	correct_path_positions = os.path.join(rawPath, 'Segmented Movements', 'Kinect', 'Positions')
	correct_path_angles = os.path.join(rawPath, 'Segmented Movements', 'Kinect', 'Angles')
	for filename in os.listdir(correct_path_positions):
		if filename.endswith('_positions.txt'):
			data = pd.read_csv(os.path.join(correct_path_positions, filename), delimiter=",")
			numpy_data_positions = data.to_numpy()

# Replace 'positions' with 'angles' in the filename and load the corresponding angles file
			filename_angles = filename.replace('positions', 'angles')
			data = pd.read_csv(os.path.join(correct_path_angles, filename_angles), delimiter=",")
			numpy_data_angles = data.to_numpy()

# Ensure both position and angle data are of the same shape
			assert numpy_data_positions.shape == numpy_data_angles.shape, "Data shapes do not match!"

# Convert both data arrays into the format Mx22x3
			numpy_data_positions = numpy_data_positions.reshape(-1, 22, 3)
			numpy_data_angles = numpy_data_angles.reshape(-1, 22, 3)

			result = []
# Iterate over the temporal length M
			for pos_slice, angle_slice in zip(numpy_data_positions, numpy_data_angles):
# Call the preprocessing function on each corresponding 22x3 slice
				B = construct_skeleton_kinect(pos_slice, angle_slice)
				result.append(B)

# Stack the result to get an array of shape Mx22x3
			result = np.stack(result, axis=0)
			augResults = getAugList(result)
# Extract movement, subject and episode number from filename
			movement, _, _, _ = filename.split('_')
			print(filename, "1", flush=True)

			for augResult in augResults:
				prmdDataset.append((augResult, movement, 1))

	incorrect_path_positions = os.path.join(rawPath, 'Incorrect Segmented Movements', 'Kinect', 'Positions')
	incorrect_path_angles = os.path.join(rawPath, 'Incorrect Segmented Movements', 'Kinect', 'Angles')
	for filename in os.listdir(incorrect_path_positions):
		if filename.endswith('_positions_inc.txt'):
			data = pd.read_csv(os.path.join(incorrect_path_positions, filename), delimiter=",")
			numpy_data_positions = data.to_numpy()

# Replace 'positions' with 'angles' in the filename and load the corresponding angles file
			filename_angles = filename.replace('positions', 'angles')
			data = pd.read_csv(os.path.join(incorrect_path_angles, filename_angles), delimiter=",")
			numpy_data_angles = data.to_numpy()

# Ensure both position and angle data are of the same shape
			assert numpy_data_positions.shape == numpy_data_angles.shape, "Data shapes do not match!"

# Convert both data arrays into the format Mx22x3
			numpy_data_positions = numpy_data_positions.reshape(-1, 22, 3)
			numpy_data_angles = numpy_data_angles.reshape(-1, 22, 3)

			result = []
# Iterate over the temporal length M
			for pos_slice, angle_slice in zip(numpy_data_positions, numpy_data_angles):
# Call the preprocessing function on each corresponding 22x3 slice
				B = construct_skeleton_kinect(pos_slice, angle_slice)
				result.append(B)

# Stack the result to get an array of shape Mx22x3
			result = np.stack(result, axis=0)
			augResults = getAugList(result)
# Extract movement, subject and episode number from filename
			movement, _, _, _, _ = filename.split('_')
			print(filename, "0", flush=True)
			for augResult in augResults:
				prmdDataset.append((augResult, movement, 0))

	# Convert lists to numpy arrays
	prmdDataset = np.array(prmdDataset, dtype=object)
	
	# Assuming I don't need the expanded data anymore -- for now commented out because its too early
	# shutil.rmtree(rawPath + '/incprmd')
	# shutil.rmtree(rawPath + '/corprmd')
	
	# Save to pickle
	with open(cleanPath + 'prmdDataset_absDA.pkl', 'wb') as f:
		pickle.dump(prmdDataset, f)

processPRMD()

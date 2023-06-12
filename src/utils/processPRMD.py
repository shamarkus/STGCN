import numpy as np
import pandas as pd
import pickle
import zipfile
import os

def processPRMD():

	mainPath = '../../'
	rawPath = mainPath + 'raw/'
	cleanPath = mainPath + 'clean/'

	zip_files = [
		rawPath + 'corprmd.zip',
		rawPath + 'incprmd.zip'
	]

	for zip_file in zip_files:
		with zipfile.ZipFile(zip_file, 'r') as zip_ref:
			zip_ref.extractall(rawPath)

	prmdDataset = []

  # Correct Segmented Movements
	correct_path = os.path.join(rawPath, 'corprmd', 'Kinect', 'Positions')
	for filename in os.listdir(correct_path):
		if filename.endswith('_positions.txt'):
			data = pd.read_csv(os.path.join(correct_path, filename), delimiter=",")
			numpy_data = data.to_numpy()
			# Extract movement, subject and episode number from filename
			movement, _, _, _ = filename.split('_')
			prmdDataset.append((numpy_data, movement, 1))

	# Incorrect Segmented Movements
	incorrect_path = os.path.join(rawPath, 'incprmd', 'Kinect', 'Positions')
	for filename in os.listdir(incorrect_path):
		if filename.endswith('_positions_inc.txt'):
			data = pd.read_csv(os.path.join(incorrect_path, filename), delimiter=",")
			numpy_data = data.to_numpy()
			# Extract movement, subject and episode number from filename
			movement, _, _, _, _ = filename.split('_')
			prmdDataset.append((numpy_data, movement, 0))

	# Convert lists to numpy arrays
	prmdDataset = np.array(prmdDataset, dtype=object)
	
	# Assuming I don't need the expanded data anymore -- for now commented out because its too early
	# shutil.rmtree(rawPath + '/incprmd')
	# shutil.rmtree(rawPath + '/corprmd')
	
	# Save to pickle
	with open(cleanPath + 'prmdDataset.pkl', 'wb') as f:
		pickle.dump(prmdDataset, f)

processPRMD()

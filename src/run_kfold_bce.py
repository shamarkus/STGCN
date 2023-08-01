import argparse
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from training import trainModel_bce
from training.utils import create_adjacency_matrix, getDataset, collate_fn
from scipy.spatial.transform import Rotation as R
from models import *

# def __RI__(sample):
# 	return np.einsum('ijk,ilk->ijl', sample, sample)

def __augment__(sample, aug_angle):
	euler = (2 * np.random.rand(3) - 1) * (aug_angle / 180.) * np.pi
	rotation = R.from_euler('zxy', euler, degrees=False).as_matrix()
	sample = np.einsum('ij,kmj->kmi', rotation, sample)
	return sample

def getAugDataset(dataset):
	old_len = len(dataset)

	for i in range(old_len):
		sample = dataset.data[i]
		for angle in [1, 2, 5, 10]:
			aug_sample = __augment__(sample[0],angle)
			dataset.append([[aug_sample, sample[1], sample[2]]])

	return dataset

# def getRIDataset(dataset):
# 
# 	for i in range(len(dataset)):
# 		dataset.prmd_data[i][0] = __RI__(dataset.prmd_data[i][0])
# 
# 	return dataset

# Define and parse command-line arguments
def run():
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('--dataset', type=str, help='Dataset')
	parser.add_argument('--model_str', type=str, help='Model String')
	parser.add_argument('--pixel_dimension', type=int, help='Pixel dimension')
	parser.add_argument('--joint_dimension', type=int, help='Joint dimension')
	parser.add_argument('--output_dimension', type=int, help='Output dimension')
	parser.add_argument('--learning_rate', type=float, help='Learning rate')
	parser.add_argument('--num_epochs', type=int, help='Number of epochs')
	parser.add_argument('--batch_size', type=int, help='Batch size')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a model
	print(device, flush=True)
	A = create_adjacency_matrix(args.joint_dimension)
	if args.model_str == '9B':
		model = STGCN_9B(args.pixel_dimension, args.output_dimension, A).to(device)
	elif args.model_str == '3B':
		model = STGCN_3B(args.pixel_dimension, args.output_dimension, A).to(device)
	elif args.model_str == '3B_RI':
		model = STGCN_3B_RI(args.pixel_dimension, args.output_dimension, A).to(device)

	# KFOLD cross validation - Extract labels from the dataset
	exercise_dataset = getDataset(args.dataset)
	labels = [exercise_dataset[i][1] for i in range(len(exercise_dataset))]

	skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

	for fold, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(exercise_dataset)), labels)):

		# For the purposes of augmenting or novelties
		train_dataset = exercise_dataset.get_subset(train_indices)
		val_dataset = exercise_dataset.get_subset(val_indices)

		if "DA" in args.dataset:
			print("Augmenting Data...")
			train_dataset = getAugDataset(train_dataset)
		
# 		if "RI" in args.dataset:
# 			print("Rotation Invariance Novelty...")
# 			train_dataset = getRIDataset(train_dataset)

		print(len(train_dataset), len(val_dataset), len(exercise_dataset))
		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True, collate_fn=collate_fn)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle = True, collate_fn=collate_fn)

		print(f"------- Training model on exercise = {args.dataset}, fold = {fold}, lr = {args.learning_rate}, bs = {args.batch_size}", flush=True)
		global_training_losses, global_validation_losses = trainModel_bce(model, args.model_str, args.dataset, device, args.learning_rate, args.batch_size, args.num_epochs, train_loader, val_loader, fold)

		# Reset the model
		if args.model_str == '9B':
			model = STGCN_9B(args.pixel_dimension, args.output_dimension, A).to(device)
		elif args.model_str == '3B':
			model = STGCN_3B(args.pixel_dimension, args.output_dimension, A).to(device)
		elif args.model_str == '3B_RI':
			model = STGCN_3B_RI(args.pixel_dimension, args.output_dimension, A).to(device)

# python run.py --dataset m01 --pixel_dimension 3 --joint_dimension 22 --output_dimension 128 --learning_rate 0.0001 --num_epochs 10000 --batch_size 1
if __name__ == '__main__':
	run()


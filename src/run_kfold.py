import argparse
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from training import trainModel
from training.utils import create_adjacency_matrix, getDataset, collate_fn
from models import *

# Define and parse command-line arguments
def run():
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('--dataset', type=str, help='Dataset')
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
	model = STGCN(args.pixel_dimension, args.output_dimension, A).to(device)

	# KFOLD cross validation - Extract labels from the dataset
	exercise_dataset = getDataset(args.dataset)
	labels = [exercise_dataset[i][1] for i in range(len(exercise_dataset))]

	skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

	for fold, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(exercise_dataset)), labels)):

		train_dataset = Subset(exercise_dataset, train_indices)
		val_dataset = Subset(exercise_dataset, val_indices)

		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True, collate_fn=collate_fn)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle = True, collate_fn=collate_fn)

		print(f"------- Training model on exercise = {args.dataset}, fold = {fold}, lr = {args.learning_rate}, bs = {args.batch_size}", flush=True)
		global_training_losses, global_validation_losses = trainModel(model, args.dataset, device, args.learning_rate, args.batch_size, args.num_epochs, train_loader, val_loader, fold)

		# Reset the model
		model = STGCN(args.pixel_dimension, args.output_dimension, A).to(device)

# python run.py --dataset m01 --pixel_dimension 3 --joint_dimension 22 --output_dimension 128 --learning_rate 0.0001 --num_epochs 10000 --batch_size 1
if __name__ == '__main__':
	run()


import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr

from training.utils import create_adjacency_matrix, getDataset
from models import *

def find_best_models(models_directory):
	best_models = {}
	for filename in os.listdir(models_directory):
		if filename.endswith(".pth"):
# ABS
			exercise, _, fold, _, _, val_loss, _ , _ = filename.split('_')
			# exercise, _, fold, _, _, val_loss, _ = filename.split('_')
			# exercise, fold, _, _, val_loss, _ = filename.split('_')
			val_loss = float(val_loss[1:])
		# 	Es3_FOLD2_E441_T127.1582_V152.9364_3B.pth
		# 	Es3_RI_FOLD2_E441_T150.9523_V85.5333_3B_RI.pth
# RI or DA
			if exercise not in best_models:
				best_models[exercise] = {}
			if fold not in best_models[exercise] or val_loss < best_models[exercise][fold][1]:
				best_models[exercise][fold] = (filename, val_loss)
	return best_models

# Function for calculating MAE and MSE
# Add spearmans correlation
def calculate_metrics(predictions, truths):
	predictions = np.concatenate(np.array(predictions))
	truths = np.ravel(truths)

	mse = mean_squared_error(truths, predictions)
	mae = mean_absolute_error(truths, predictions)
	spearman_corr, _ = spearmanr(truths, predictions)
	return mse, mae, spearman_corr

# Function for generating scatter plot
def generate_scatter_plot(fold_predictions, fold_truths, title):
	if fold_predictions and fold_truths:
		plt.figure(figsize=(15, 10)) # Increased figure width

# Concatenate all predictions and truths across folds
		all_predictions = np.concatenate(fold_predictions)
		all_truths = np.concatenate(fold_truths)

# Get a sorted list of indices based on the truths
		sorted_indices = np.argsort(all_truths)

# Sort predictions and truths according to these indices
		all_predictions = all_predictions[sorted_indices]
		all_truths = all_truths[sorted_indices]

		mean_truths = np.mean(all_truths)

		plt.axhline(y=mean_truths, color='blue', linestyle='--', label='Mean Truth')

# Plot truths (labels) and predictions with thicker, larger markers
		plt.scatter(np.arange(len(all_truths)), all_truths, color='black', edgecolor='black', facecolor='none', linewidth=2, s=100, marker='s', label='Label')
		plt.scatter(np.arange(len(all_predictions)), all_predictions, color='red', edgecolor='red', facecolor='none', linewidth=2, s=100, marker='o', label='Prediction')

		plt.xlabel('Sorted Sample Indices')
		plt.ylabel('Values')
		plt.title(title)
		plt.legend()
		plt.grid(True)
		plt.savefig(f"{title}.png")

def get_predictions(model, dataloader, device):
	model.eval()
	predictions = []
	truths = []

	with torch.no_grad():
		for joint_positions, label in dataloader:
			# Move tensors to the configured device
			joint_positions = joint_positions.to(device)
			label = label.to(device).view(-1, 1)
			# Forward pass
			outputs = model(joint_positions, [])

			# Collect the predictions and the true labels
			predictions.append(outputs.item())
			truths.append(label.item())

	return predictions, truths

def analysis():
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('--model_str', type=str, help='Model String')
	parser.add_argument('--pixel_dimension', type=int, help='Pixel dimension')
	parser.add_argument('--joint_dimension', type=int, help='Joint dimension')
	parser.add_argument('--output_dimension', type=int, help='Output dimension')
	args = parser.parse_args()

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	models_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/ES3_ABSDARI/'))
	best_models = find_best_models(models_directory)

	skf = KFold(n_splits=5, random_state=1, shuffle=True)

	# 22 for UIPRMD, 25 for KIMORE
	A = create_adjacency_matrix(args.joint_dimension)

	for exercise, exercise_data in best_models.items():
		# RI OR DA
		# exercise_dataset = getDataset(exercise + '_DA')
# ABS
		exercise_dataset = getDataset(exercise)
		labels = [exercise_dataset[i][1] for i in range(len(exercise_dataset))]
		fold_predictions = []
		fold_truths = []

		for fold, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(exercise_dataset)), labels)):
			if (f"FOLD{fold}") in exercise_data:
				val_dataset = Subset(exercise_dataset, val_indices)
				val_loader = DataLoader(val_dataset, 1)

				# Model Dependent
				if args.model_str == '9B':
					model = STGCN_9B(args.pixel_dimension, args.output_dimension, A).to(device)
				elif args.model_str == '3B':
					model = STGCN_3B(args.pixel_dimension, args.output_dimension, A).to(device)
				elif args.model_str == '3B_RI':
					model = STGCN_3B_RI(args.pixel_dimension, args.output_dimension, A).to(device)

				model_path = os.path.join(models_directory, exercise_data[f"FOLD{fold}"][0])
				model.load_state_dict(torch.load(model_path, map_location=device))

				print(model_path)

				predictions, truths = get_predictions(model, val_loader, device) # Make this function

				fold_predictions.append(predictions)
				fold_truths.append(truths)

		mse, mae, spearman_corr = calculate_metrics(fold_predictions, fold_truths)
		print(f"Exericse: {exercise}, MSE: {mse}, MAE: {mae}, spearman_corr: {spearman_corr}")

		generate_scatter_plot(fold_predictions, fold_truths, f"{exercise}")

if __name__ == '__main__':
	analysis()

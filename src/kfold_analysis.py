import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

from training.utils import create_adjacency_matrix, getDataset
from models import *

def find_best_models(models_directory):
	best_models = {}
	for filename in os.listdir(models_directory):
		if filename.endswith(".pth"):
			exercise, fold, _, _, val_loss, _ = filename.split('_')
			val_loss = float(val_loss[1:])
			if exercise not in best_models:
				best_models[exercise] = {}
			if fold not in best_models[exercise] or val_loss < best_models[exercise][fold][1]:
				best_models[exercise][fold] = (filename, val_loss)
	return best_models

# Function for calculating MAE and MSE
def calculate_metrics(predictions, truths):
	mse = mean_squared_error(truths, predictions)
	mae = mean_absolute_error(truths, predictions)
	return mse, mae

# Function for generating scatter plot
def generate_scatter_plot(fold_predictions, fold_truths, title):
	plt.figure(figsize=(10, 10))

	markers = ['o', 's']  # Different shapes for labels 0 and 1
	colors = ['b', 'g', 'r', 'c', 'm']  # Different colors for each fold

	for i, (predictions, truths) in enumerate(zip(fold_predictions, fold_truths)):
		for label in np.unique(truths):
			subset = np.where(truths == label)
			plt.scatter(np.array(predictions)[subset], np.array(truths)[subset], alpha=0.5, 
				marker=markers[int(label)], c=colors[i], label=f'Fold {i+1}, Label {int(label)}')
	plt.xlabel('Predictions')
	plt.ylabel('Ground Truth')
	plt.title(title)
	plt.legend()
	plt.grid(True)
	plt.show()

def get_predictions(model, dataloader, device):
	model.eval()
	predictions = []
	truths = []

	with torch.no_grad():
		for joint_positions, lengths, label in dataloader:
# Move tensors to the configured device
			joint_positions = joint_positions.to(device).reshape(-1, joint_positions.size(1), int(joint_positions.size(2) / 3), 3)
			label = label.to(device).view(-1, 1)

# Forward pass
			outputs = model(joint_positions, lengths)

# Collect the predictions and the true labels
			predictions.extend(outputs.squeeze().detach().cpu().numpy())
			truths.extend(label.squeeze().detach().cpu().numpy())

	return predictions, truths

def analysis():
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	models_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/'))
	best_models = find_best_models(models_directory)

	skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)

	A = create_adjacency_matrix(args.joint_dimension)

	for exercise, exercise_data in best_models.items():
		exercise_dataset = getDataset(exercise)
		labels = [exercise_dataset[i][1] for i in range(len(exercise_dataset))]
		fold_predictions = []
		fold_truths = []

		for fold, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(exercise_dataset)), labels)):
			if str(fold) in exercise_data:
				val_dataset = Subset(exercise_dataset, val_indices)
				val_loader = DataLoader(val_dataset, 1)

				model = STGCN(3, 128, A).to(device)
				model_path = os.path.join(models_directory, exercise_data[str(fold)][0])
				model.load_state_dict(torch.load(model_path))
				model.eval()

				predictions, truths = get_predictions(model, val_loader, device) # Make this function

				mse, mae = calculate_metrics(predictions, truths)
				print(f"Exercise: {exercise}, Fold: {fold}, MSE: {mse}, MAE: {mae}")

				fold_predictions.append(predictions)
				fold_truths.append(truths)

		generate_scatter_plot(fold_predictions, fold_truths, f"Exercise: {exercise}")

if __name__ == '__main__':
	analysis()

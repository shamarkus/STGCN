import argparse
import torch
from training import trainModel
from training.utils import create_adjacency_matrix
from models import *

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

# Train and Evaluate
	global_training_losses, global_validation_losses = trainModel(model, args.model_str, args.dataset, device, args.learning_rate, args.batch_size, args.num_epochs)

# python run.py --dataset m01 --pixel_dimension 3 --joint_dimension 22 --output_dimension 128 --learning_rate 0.0001 --num_epochs 10000 --batch_size 1
if __name__ == '__main__':
	run()


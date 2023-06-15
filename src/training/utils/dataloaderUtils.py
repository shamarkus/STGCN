import random
import pickle
import torch
import os

from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

from .datasets import *

def getDataset(es_str):
	# Get the absolute path to the data directory
	clean_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/clean'))

	# Specify the path to the pickle file
	if es_str in ['m01', 'm02', 'm03', 'm04', 'm05', 'm06', 'm07', 'm08', 'm09', 'm10']:
		pickle_path = os.path.join(clean_path, 'prmdDataset.pkl')
		dataset_class = PRMDCustomDataset
	elif es_str in ['Es1', 'Es2', 'Es3', 'Es4', 'Es5']:
		pickle_path = os.path.join(clean_path, 'kimoreDataset.pkl')
		dataset_class = KimoreCustomDataset
	else:
		raise ValueError('Invalid es_str parameter.')

	# Read the data from the pickle file
	with open(pickle_path, 'rb') as f:
		dataset = pickle.load(f)

	return dataset_class(dataset).filter_by_exercise(exercise=es_str)

def collate_fn(batch):
	# Sort the batch in the descending order of sequence lengths
	batch.sort(key=lambda x: x[0].shape[0], reverse=True)

	# Separate the sequences and the labels
	sequences, labels = zip(*batch)

	# Get the lengths of sequences
	lengths = [seq.shape[0] for seq in sequences]

	# Pad the sequences
	sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

	return sequences_padded, lengths, torch.stack(labels)

def getDataloaders(es_str, batch_size, random_seed = 3407):
	random.seed(random_seed)
	torch.manual_seed(random_seed)

	# Read the exercise dataset
	exercise_dataset = getDataset(es_str)

	num_samples = len(exercise_dataset)
	num_val_samples = int(0.2 * num_samples)
	num_train_samples = num_samples - num_val_samples

	# Create a random split using the random_split function
	train_dataset, val_dataset = random_split(exercise_dataset, [num_train_samples, num_val_samples])

	return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn), DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

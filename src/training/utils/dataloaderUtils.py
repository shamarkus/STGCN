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
	elif es_str in ['m01_abs', 'm02_abs', 'm03_abs', 'm04_abs', 'm05_abs', 'm06_abs', 'm07_abs', 'm08_abs', 'm09_abs', 'm10_abs']:
		pickle_path = os.path.join(clean_path, 'prmdDataset_abs.pkl')
		dataset_class = PRMDCustomDataset
	elif es_str in ['m01_absRI', 'm02_absRI', 'm03_absRI', 'm04_absRI', 'm05_absRI', 'm06_absRI', 'm07_absRI', 'm08_absRI', 'm09_absRI', 'm10_absRI']:
		pickle_path = os.path.join(clean_path, 'prmdDataset_abs.pkl')
		dataset_class = PRMDCustomDataset
	elif es_str in ['m01_absDA', 'm02_absDA', 'm03_absDA', 'm04_absDA', 'm05_absDA', 'm06_absDA', 'm07_absDA', 'm08_absDA', 'm09_absDA', 'm10_absDA']:
		# For the purposes of a non-augmented dataset, we need to use the same dataset, but augment it within trainModel
		pickle_path = os.path.join(clean_path, 'prmdDataset_abs.pkl')
		# pickle_path = os.path.join(clean_path, 'prmdDataset_absDA.pkl')
		dataset_class = PRMDCustomDataset
	elif es_str in ['m01_absDARI', 'm02_absDARI', 'm03_absDARI', 'm04_absDARI', 'm05_absDARI', 'm06_absDARI', 'm07_absDARI', 'm08_absDARI', 'm09_absDARI', 'm10_absDARI']:
		# For the purposes of a non-augmented dataset, we need to use the same dataset, but augment it within trainModel
		pickle_path = os.path.join(clean_path, 'prmdDataset_abs.pkl')
		# pickle_path = os.path.join(clean_path, 'prmdDataset_absDARI.pkl')
		dataset_class = PRMDCustomDataset
	# KiMoRe section
	elif es_str in ['Es1', 'Es2', 'Es3', 'Es4', 'Es5', 'Es1_RI', 'Es2_RI', 'Es3_RI', 'Es4_RI', 'Es5_RI', 'Es1_DA', 'Es2_DA', 'Es3_DA', 'Es4_DA', 'Es5_DA', 'Es1_DARI', 'Es2_DARI', 'Es3_DARI', 'Es4_DARI', 'Es5_DARI']:
		pickle_path = os.path.join(clean_path, 'kimoreDataset.pkl')
		dataset_class = KimoreCustomDataset
	elif es_str in ['I0', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I0_RI', 'I1_RI', 'I2_RI', 'I3_RI', 'I4_RI', 'I5_RI', 'I6_RI', 'I7_RI', 'I8_RI',  'I0_DA', 'I1_DA', 'I2_DA', 'I3_DA', 'I4_DA', 'I5_DA', 'I6_DA', 'I7_DA', 'I8_DA', 'I0_DARI', 'I1_DARI', 'I2_DARI', 'I3_DARI', 'I4_DARI', 'I5_DARI', 'I6_DARI', 'I7_DARI', 'I8_DARI']:
		pickle_path = os.path.join(clean_path, 'intellirehab.pkl')
		dataset_class = IntellirehabCustomDataset
	else:
		raise ValueError('Invalid es_str parameter.')

	# Read the data from the pickle file
	with open(pickle_path, 'rb') as f:
		dataset = pickle.load(f)

	return dataset_class(dataset).filter_by_exercise(exercise=es_str.split('_')[0])

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

def getDataloaders(es_str, batch_size, random_seed = 3770):
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

import torch
import numpy as np
from torch.utils.data import Dataset

class KimoreCustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        joint_positions = self.data[idx][0]
        joint_positions = np.delete(joint_positions, np.arange(3, joint_positions.shape[1], 4), axis=1)  # remove every 4th element starting from index 3
        joint_positions = joint_positions[:,:-1] 
        joint_positions = torch.tensor(joint_positions, dtype=torch.float32)
        data_class = torch.tensor(self.data[idx][2], dtype=torch.long)
        label = torch.tensor(self.data[idx][3], dtype=torch.float32)

        return joint_positions, label

    def filter_by_exercise(self, exercise):
        filtered_data = [data for data in self.data if data[4] == exercise]
        return KimoreCustomDataset(filtered_data)

# # Assuming 'data' is the numpy array containing the data
# dataset = KimoreCustomDataset(kimoreDataset)
# 
# # Filter dataset based on exercise 'Es1'
# ES1_dataset = dataset.filter_by_exercise(exercise='Es1')
# ES2_dataset = dataset.filter_by_exercise(exercise='Es2')
# ES3_dataset = dataset.filter_by_exercise(exercise='Es3')
# ES4_dataset = dataset.filter_by_exercise(exercise='Es4')
# ES5_dataset = dataset.filter_by_exercise(exercise='Es5')

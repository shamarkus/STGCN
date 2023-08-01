import torch
import numpy as np
from torch.utils.data import Dataset

class KimoreCustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        joint_positions, movement, label = self.data[idx]

        joint_positions = torch.tensor(joint_positions, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return joint_positions, label

    def filter_by_exercise(self, exercise):
        for i in self.data[:, 0:1]:
            i[0] = i[0][:, :-1].reshape((-1,25,4))[:,:,:3]

        self.data = np.concatenate((self.data[:, 0:1], self.data[:, 4:5], self.data[:, 3:4]), axis=1)
        filtered_data = [data for data in self.data if data[1] == exercise]
        return KimoreCustomDataset(filtered_data)

    def get_subset(self, indices):
        return KimoreCustomDataset([self.data[i] for i in indices])

    def append(self, new_data):
        self.data.extend(new_data)

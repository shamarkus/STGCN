import torch
from torch.utils.data import Dataset

class PRMDCustomDataset(Dataset):
    def __init__(self, prmd_data):
        self.prmd_data = prmd_data

    def __len__(self):
        return len(self.prmd_data)

    def __getitem__(self, idx):
        joint_positions, movement, label = self.prmd_data[idx]
        # Convert data to PyTorch tensors and label to float32
        joint_positions = torch.tensor(joint_positions, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return joint_positions, label

    def filter_by_exercise(self, exercise):
        # Filter the data based on the movement type/number
        filtered_data = [data for data in self.prmd_data if data[1] == exercise]
        return PRMDCustomDataset(filtered_data) 

    def get_subset(self, indices):
        return PRMDCustomDataset([self.prmd_data[i] for i in indices])

    def append(self, new_data):
        self.prmd_data.extend(new_data)


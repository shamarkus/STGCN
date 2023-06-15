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

# # Assuming 'data' is the numpy array containing the data
# dataset = PRMDCustomDataset(prmdDataset)
# 
# # Filter dataset based on exercise 'Es1'
# m01_dataset = dataset.filter_by_exercise(exercise='m01')
# m02_dataset = dataset.filter_by_exercise(exercise='m02')
# m03_dataset = dataset.filter_by_exercise(exercise='m03')
# m04_dataset = dataset.filter_by_exercise(exercise='m04')
# m05_dataset = dataset.filter_by_exercise(exercise='m05')
# m06_dataset = dataset.filter_by_exercise(exercise='m06')
# m07_dataset = dataset.filter_by_exercise(exercise='m07')
# m08_dataset = dataset.filter_by_exercise(exercise='m08')
# m09_dataset = dataset.filter_by_exercise(exercise='m09')
# m10_dataset = dataset.filter_by_exercise(exercise='m10')

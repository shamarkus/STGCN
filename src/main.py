from google.colab import drive
from torch import nn, Tensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split
from torch_geometric.nn import GCNConv
import shutil
import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
import zipfile
import os
import sys
import pandas as pd
import numpy as np
import warnings
import random
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")
mainPath = '../../'
rawPath = mainPath + 'raw/'
cleanPath = mainPath + 'clean/'

zip_files = [
    rawPath + 'corprmd.zip',
    rawPath + 'incprmd.zip'
]

for zip_file in zip_files:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(rawPath)

prmdDataset = []

  # Correct Segmented Movements
correct_path = os.path.join(rawPath, 'corprmd', 'Kinect', 'Positions')
for filename in os.listdir(correct_path):
    if filename.endswith('_positions.txt'):
        data = pd.read_csv(os.path.join(correct_path, filename), delimiter=",")
        numpy_data = data.to_numpy()
        # Extract movement, subject and episode number from filename
        movement, _, _, _ = filename.split('_')
        prmdDataset.append((numpy_data, movement, 1))

# Incorrect Segmented Movements
incorrect_path = os.path.join(rawPath, 'incprmd', 'Kinect', 'Positions')
for filename in os.listdir(incorrect_path):
    if filename.endswith('_positions_inc.txt'):
        data = pd.read_csv(os.path.join(incorrect_path, filename), delimiter=",")
        numpy_data = data.to_numpy()
        # Extract movement, subject and episode number from filename
        movement, _, _, _, _ = filename.split('_')
        prmdDataset.append((numpy_data, movement, 0))

# Convert lists to numpy arrays
prmdDataset = np.array(prmdDataset, dtype=object)

# Assuming I don't need the expanded data anymore -- for now commented out because its too early
# shutil.rmtree(rawPath + '/incprmd')
# shutil.rmtree(rawPath + '/corprmd')

# Save to pickle
with open(cleanPath + 'prmdDataset.pkl', 'wb') as f:
    pickle.dump(prmdDataset, f)





kimorePath = mainPath + '/KIMORE/KiMoRe.zip'

with zipfile.ZipFile(kimorePath, 'r') as zip_ref:
    zip_ref.extractall('./')

# Define directory structure
base_dir = './KiMoRe'
class_dir = ['CG', 'GPP']
sub_class_dir = [['Expert', 'NotExpert'], ['BackPain', 'Parkinson', 'Stroke']]
exercises = ['Es1', 'Es2', 'Es3', 'Es4', 'Es5']

# Define class mapping
class_mapping = {'CG-Expert': 0, 'CG-NotExpert': 1, 'GPP-Stroke': 2, 'GPP-Parkinson': 3, 'GPP-BackPain': 4}

# Define lists to store data
data_list = []

# Loop through all directories and subdirectories
for class_idx, class_name in enumerate(class_dir):
    for sub_class_name in sub_class_dir[class_idx]:
        individuals_dir = os.path.join(base_dir, class_name, sub_class_name)
        individuals = os.listdir(individuals_dir)

        for individual in individuals:
            individual_dir = os.path.join(individuals_dir, individual)
            for exercise in exercises:
                exercise_dir = os.path.join(individual_dir, exercise)

                # Load labels
                labels_path = os.path.join(exercise_dir, 'Label', f'ClinicalAssessment_{individual}.xlsx')
                label_df = pd.read_excel(labels_path)
                labels = label_df.iloc[0, 1:6].values
                label = labels[int(exercise.replace('Es', '')) - 1]

                # Load data
                raw_dir = os.path.join(exercise_dir, 'Raw')
                files = os.listdir(raw_dir)

                data_dict = {'JointPosition': np.array([]), 'JointOrientation': np.array([])}  # Initialize data dict

                for file in files:
                    for key in data_dict.keys():
                        if key in file:
                            data_path = os.path.join(raw_dir, file)
                            try:
                                data_df = pd.read_csv(data_path, header=None)
                                
                                # If the DataFrame is not empty, convert it to a numpy array
                                if not data_df.empty:
                                    data_dict[key] = data_df.values
                            except Exception as e:
                                print(f"Error reading file {file}: {str(e)}")
                # Associate data with class
                if data_dict['JointPosition'].size != 0 and data_dict['JointOrientation'].size != 0 and not np.isnan(label):
                  data_class = class_mapping[f'{class_name}-{sub_class_name}']
                  data_list.append((data_dict['JointPosition'], data_dict['JointOrientation'], data_class, label, exercise))

# Convert lists to numpy arrays
kimoreDataset = np.array(data_list, dtype=object)

# Now, 'data' is a numpy array where each element is a tuple.
# The first element of the tuple is the sensor data for one exercise of one individual, and the second element is the class of the data.
# 'labels' is a numpy array where each element is the evaluation scores for the five exercises of one individual.


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

# Assuming 'data' is the numpy array containing the data
dataset = PRMDCustomDataset(prmdDataset)

# Filter dataset based on exercise 'Es1'
m01_dataset = dataset.filter_by_exercise(exercise='m01')
m02_dataset = dataset.filter_by_exercise(exercise='m02')
m03_dataset = dataset.filter_by_exercise(exercise='m03')
m04_dataset = dataset.filter_by_exercise(exercise='m04')
m05_dataset = dataset.filter_by_exercise(exercise='m05')
m06_dataset = dataset.filter_by_exercise(exercise='m06')
m07_dataset = dataset.filter_by_exercise(exercise='m07')
m08_dataset = dataset.filter_by_exercise(exercise='m08')
m09_dataset = dataset.filter_by_exercise(exercise='m09')
m10_dataset = dataset.filter_by_exercise(exercise='m10')

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

# Assuming 'data' is the numpy array containing the data
dataset = KimoreCustomDataset(kimoreDataset)

# Filter dataset based on exercise 'Es1'
ES1_dataset = dataset.filter_by_exercise(exercise='Es1')
ES2_dataset = dataset.filter_by_exercise(exercise='Es2')
ES3_dataset = dataset.filter_by_exercise(exercise='Es3')
ES4_dataset = dataset.filter_by_exercise(exercise='Es4')
ES5_dataset = dataset.filter_by_exercise(exercise='Es5')

# Set a random seed for reproducible training, valid, and test datasets
def getDataloaders(exercise_dataset, batch_size, random_seed = 3407):
  random.seed(random_seed)
  torch.manual_seed(random_seed)

  num_samples = len(exercise_dataset)
  num_val_samples = int(0.2 * num_samples)
  num_train_samples = num_samples - num_val_samples
  
  # Create a random split using the random_split function
  train_dataset, val_dataset = random_split(exercise_dataset, [num_train_samples, num_val_samples])

  return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn), DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

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

#  The joints in the skeletal model recorded with the Kinect sensor are shown in Figure 2. The data
#  include the motion measurement for 22 joints. The positions of the fingers are not included because
#  they are not relevant for assessing correct performance of the movements included in the study.
#  The order of the joint measurements in the data set is displayed in the figure, where the first three
#  measurements pertain to the waist joint, the next three values are for the spine, etc. The values for the
#  waist joint are given in absolute coordinates with respect to the frame of the coordinate origin of the
#  Kinect sensor, and the values for the other 21 joints are given in relative coordinates with respect to
#  the parent joint in the skeletal model. For instance, the position and orientation of the left forearm is
#  given relative to the position and orientation of the left upper a 
#  ---- IS IT FAIR TO JUST NAIVELY USE THIS DATASET, OR SHOULD RELATIVE COORDINATES BE PUT INTO PLACE
def create_prmd_adjacency_matrix():
  # initialize 22x22 matrix with zeros
  adj_matrix = np.zeros((22, 22))
  
  # define the connections
  connections = [
      (1, 2), (1, 15), (1, 19),
      (2, 1), (2, 3),
      (3, 2), (3, 4),
      (4, 3), (4, 5), (4, 7), (4, 11),
      (5, 4), (5, 6),
      (6, 5),
      (7, 4), (7, 8),
      (8, 7), (8, 9),
      (9, 8), (9, 10),
      (10, 9),
      (11, 4), (11, 12),
      (12, 11), (12, 13),
      (13, 12), (13, 14),
      (14, 13),
      (15, 1), (15, 16),
      (16, 15), (16, 17),
      (17, 16), (17, 18),
      (18, 17),
      (19, 1), (19, 20),
      (20, 19), (20, 21),
      (21, 20), (21, 22),
      (22, 21),
  ]
  
  # fill in the adjacency matrix
  for i, j in connections:
      adj_matrix[i-1, j-1] = 1
      adj_matrix[j-1, i-1] = 1  # because it's an undirected graph
  return adj_matrix

def create_adjacency_matrix(joint_dimension):
  neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                    (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                    (22, 23), (23, 8), (24, 25), (25, 12)]
  
  neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
  edge = neighbor_link
  A = np.zeros((joint_dimension, joint_dimension)) # adjacency matrix
  for i, j in edge:
      if i < joint_dimension and j < joint_dimension:  
          A[j, i] = 1
          A[i, j] = 1
  return A

def create_kimore_adjacency_matrix():
    num_joints = 25

    # Define edges based on Kinect v2 Sensor Skeleton hierarchy
    edges = [
        (1, 2),
        (2, 3),
        (3, 4), (3, 5),
        (4, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10), (8, 12),
        (9, 11), (9, 13),
        (12, 14),
        (13, 15),
        (16, 3), (16, 17),
        (17, 18), (17, 19),
        (18, 20),
        (19, 21),
        (20, 22),
        (21, 23),
        (22, 24),
        (23, 25),
    ]

    adjacency_matrix = np.zeros((num_joints, num_joints))

    for edge in edges:
        # Adjust for 0-based index
        i, j = edge[0] - 1, edge[1] - 1
        adjacency_matrix[i][j] = 1
        adjacency_matrix[j][i] = 1  # Undirected graph, symmetric adjacency matrix

    return adjacency_matrix

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=(1, 1), residual=False):
        super(STGCNBlock, self).__init__()

        self.spatial_gc = GraphConvolution(in_channels, out_channels)
        self.tcn = nn.Sequential(
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=stride),
            # nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == (1, 1)):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride),
                # nn.BatchNorm2d(out_channels),
            )

        self.A = torch.tensor(A,dtype=torch.float32)
        self.M = nn.Parameter(self.A + torch.eye(self.A.size(0))) # Make a copy of A and use it, don't bother with all ones so that you can see what's being paid attention to

    def forward(self, x):

        # Normalized Adjacency Matrix with edge importance
        A = self.A.to(x.device)
        A_hat = (A + torch.eye(A.size(0)).to(x.device)) * self.M  # (A + I) ⊗ M
        D_hat_inv_sqrt = torch.diag(torch.pow(A_hat.sum(1), -0.5))
        A_hat_norm = torch.matmul(torch.matmul(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)

        # Apply the residual connection, ensuring the input x is transformed to match dimensions
        res = self.residual(x.permute(0,3,1,2))

        x_gc = self.spatial_gc(x, A_hat_norm)
        x_gc = x_gc.permute(0,3,1,2)

        x_gc = self.tcn(x_gc) + res
        x_gc = x_gc.permute(0,2,3,1) # Resnet Mechanism

        return F.relu(x_gc)

class STGCN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim, num_layers, A):
        super(STGCN, self).__init__()

        self.pixel_dimension = in_channels
        self.joint_dimension = A.shape[0]
        self.output_dimension = out_channels
        self.hidden_dim = hidden_dim 
        self.num_layers = num_layers

        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, 32 , A, residual = False),
            STGCNBlock(32, 32, A),
            STGCNBlock(32, 32, A),
            STGCNBlock(32, 64, A, stride=(2, 1)),  # Set stride to 2 for 4th layer
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=(2, 1)),  # Set stride to 2 for 7th layer
            STGCNBlock(128, 128, A),
            STGCNBlock(128, out_channels, A),
        )

        self.lstm = nn.LSTM(out_channels * A.shape[0], hidden_dim, num_layers=num_layers, batch_first=True)  # LSTM layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),  # Linear layer
            nn.ReLU(),
            nn.Linear(32,1)
        )

    def forward(self, x, lengths):
      for i, stgcn_block in enumerate(self.stgcn_blocks):
            x = stgcn_block(x)
            if i in {3, 6}:  # Update lengths at 4th and 7th layers
                lengths = self.calculate_lengths_after_conv(lengths, kernel_size=9, stride=2, padding=4)

        
      # Reshape from (bs, M, joint_dimension, out_channels) to (bs, M, joint_dimension * out_channels)
      x = x.reshape(x.size(0), x.size(1), x.size(2) * x.size(3))
    
      # Pack the sequences - This returns a PackedSequence object instance
      x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

      # Initialize hidden state and cell state
      h0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_dim).to(x.data.device)
      c0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_dim).to(x.data.device)

      # Pass through LSTM
      _, (h0, _) = self.lstm(x, (h0, c0))

      h0 = h0[-1]

      # Pass the final hidden state through the rest of the classifier
      return self.classifier(h0)

    def calculate_lengths_after_conv(self, lengths, kernel_size, stride, padding):
      return [(length + 2*padding - (kernel_size - 1) - 1) // stride + 1 for length in lengths]
    

class VANILLA_STGCN(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(VANILLA_STGCN, self).__init__()

        self.pixel_dimension = in_channels
        self.joint_dimension = A.shape[0]
        self.output_dimension = out_channels

        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, 32 , A, residual = False),
            STGCNBlock(32, 32, A),
            STGCNBlock(32, 32, A),
            STGCNBlock(32, 64, A, stride=(1, 1)),  # Set stride to 2 for 4th layer
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=(1, 1)),  # Set stride to 2 for 7th layer
            STGCNBlock(128, 128, A),
            STGCNBlock(128, out_channels, A),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Flatten(),
            nn.Linear(self.joint_dimension * out_channels, 32),  # Linear layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32,1)
        )

    def forward(self, x, lengths):
      for i, stgcn_block in enumerate(self.stgcn_blocks):
            x = stgcn_block(x)
            if i in {3, 6}:  # Update lengths at 4th and 7th layers
                lengths = self.calculate_lengths_after_conv(lengths, kernel_size=9, stride=2, padding=4)

      return self.classifier(x.permute(0,3,1,2))
    def debug(self, x, lengths):
        outputs = []
        for i, stgcn_block in enumerate(self.stgcn_blocks):
            x = stgcn_block(x)
            outputs.append(x.detach().cpu().numpy())  # Save the output of each block, detaching it from the computation graph and converting to numpy for easier inspection
            if i in {3, 6}:  # Update lengths at 4th and 7th layers
                lengths = self.calculate_lengths_after_conv(lengths, kernel_size=9, stride=2, padding=4)

        final_output = self.classifier(x.permute(0,3,1,2))
        return final_output.detach().cpu().numpy(), outputs  # Return final output and intermediate outputs

    def calculate_lengths_after_conv(self, lengths, kernel_size, stride, padding):
      return [(length + 2*padding - (kernel_size - 1) - 1) // stride + 1 for length in lengths]
    

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for joint_positions, lengths, label in dataloader:
            # Move tensors to the configured device
            joint_positions = joint_positions.to(device).reshape(-1, joint_positions.size(1), int(joint_positions.size(2) / 3), 3)
            label = label.to(device)

            # Forward pass
            outputs = model(joint_positions, lengths)
           
            l1_lambda = 0.00
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = criterion(outputs, label) + l1_lambda * l1_norm

            running_loss += loss.item() * joint_positions.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for joint_positions, lengths, label in dataloader:
        # Move tensors to the configured device
        joint_positions = joint_positions.to(device).reshape(-1, joint_positions.size(1), int(joint_positions.size(2) / 3), 3)
        label = label.to(device)

        # Forward pass
        outputs = model(joint_positions, lengths)

        if torch.isnan(outputs[0][0]):
          sys.exit()

        l1_lambda = 0.00
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = criterion(outputs, label) + l1_lambda * l1_norm


        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

         # Clone parameters before optimizer.step()
         # params_before_update = [param.clone() for param in model.parameters()]
        
        optimizer.step()

        # Clone parameters after optimizer.step()
        # params_after_update = [param.clone() for param in model.parameters()]
        
        # Check if the parameters have been updated
        # parameters_updated = all(torch.equal(param_before, param_after) for param_before, param_after in zip(params_before_update, params_after_update))
        # print(f'Parameters Updated: {not parameters_updated}')

        running_loss += loss.item() * joint_positions.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


class CustomLRAdjuster:
    def __init__(self, optimizer, threshold, factor):
        self.optimizer = optimizer
        self.threshold = threshold
        self.factor = factor
        self.lr_dropped = False

    def step(self, loss):
        if loss < self.threshold and not self.lr_dropped:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.factor
            self.lr_dropped = True


global_training_losses = []
global_validation_losses = []

def trainModel(model, ES_dataset, device, lr = 0.01, bs = 3, num_epochs = 60):
  
  # Define the loss function and optimizer
  criterion = nn.MSELoss()
  # criterion = nn.L1Loss()
  optimizer = torch.optim.Adam(model.parameters(), lr = lr)

  # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.1)
  # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.2, patience = 4, threshold = 0.001)
  scheduler = CustomLRAdjuster(optimizer, threshold=0.2, factor=0.1)
  
  # Get the dataloaders
  train_loader, val_loader = getDataloaders(ES_dataset, bs)

  # Training Loop
  for epoch in range(num_epochs):
      train_loss = train(model, train_loader, criterion, optimizer, device)

      val_loss = validate(model, val_loader, criterion, device)
      #scheduler.step()
      scheduler.step(val_loss)

  
      global_training_losses.append(train_loss)
      global_validation_losses.append(val_loss)

      # Save model if validation loss is less than 0.150
      if val_loss < 0.400:
          torch.save(model.state_dict(), './best_model.pth')
      print(f"Epoch {epoch+1}/{num_epochs}.. Train loss: {train_loss:.3f}.. Validation loss: {val_loss:.3f}")

  # Hopefully this is a return by reference
  return model;

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # Create a model
  pixel_dimension = 3
  joint_dimension = 22
  output_dimension = 128 
  hidden_dim = 80
  num_layers = 4
  A = create_adjacency_matrix(joint_dimension)

  model = STGCN(pixel_dimension, output_dimension, hidden_dim, num_layers, A).to(device)

  # Train and Evaluate
  ES_dataset = m05_dataset
  model = trainModel(model, ES_dataset, device, 0.0005, 1, 1000)

  # ResNet mechanism
  # Figure out good lr/scheduler injection point -- Let's assume batch size 1 for now, since its the best
  # Get a working attention matrix mechanism - currently its very minimal

  # What loss should I train my model on -- afterwards do cross-validation

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # Create a model
  pixel_dimension = 3
  joint_dimension = 22
  output_dimension = 128 
  A = create_adjacency_matrix(joint_dimension)

  model = VANILLA_STGCN(pixel_dimension, output_dimension, A).to(device)

  # Train and Evaluate
  ES_dataset = m05_dataset
  model = trainModel(model, ES_dataset, device, 0.0001, 1, 1000)

  # ResNet mechanism
  # Figure out good lr/scheduler injection point -- Let's assume batch size 1 for now, since its the best
  # Get a working attention matrix mechanism - currently its very minimal

  # What loss should I train my model on -- afterwards do cross-validation

!jupyter nbconvert --to script main.ipynb


def plot_losses():
  plt.figure(figsize=(10,5))
  plt.plot(global_training_losses, label='Training Loss')
  plt.plot(global_validation_losses, label='Validation Loss')
  plt.title('Training and Validation Losses')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()
plot_losses()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()
train_loader, val_loader = getDataloaders(m05_dataset, 1)
for joint_positions,lengths, c in val_loader:
  joint_positions = joint_positions.to(device).reshape(-1, joint_positions.size(1), 22, 3)
  break
  # print("Predicted: ", model.debug(joint_positions, lengths))
  _, tensor_list = model.debug(joint_positions, lengths)
  
  for tensor in tensor_list:
      numpy_arr = tensor
      np.set_printoptions(threshold=sys.maxsize)  # This line ensures numpy prints the entire array
      print(numpy_arr)
      print("\n\n-------------------------")
  print("\n")
  break

# for joint_positions,lengths,c in train_loader:
#   joint_positions = joint_positions.to(device).reshape(-1, joint_positions.size(1), 22, 3)
#   print("Predicted: ", model(joint_positions, lengths).cpu().detach().numpy())
#   print("Actual:", c.cpu().detach().numpy())
#   print("\n")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_counts(model):
    print("Model Parameter Counts:")
    total_params = count_parameters(model)
    sum = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            percentage = (param_count / total_params) * 100
            sum += percentage
            print(f"{name}: {param_count} ({percentage:.2f}%) -- So far ({sum}%)")
    print(f"Total trainable parameters: {total_params}")

print_parameter_counts(model)

def check_for_exploding_or_vanishing_params(model):
    for name, param in model.named_parameters():
        if (param.data.isnan().any() or param.data.isinf().any()):
            print(f'Parameter {name} has nan or inf values')

        # Check if the parameters are vanishing
        elif param.data.abs().max() < 1e-4:
            print(f'Parameter {name} is v4nishing (max abs value less than 1e-10)')

        # Check if the parameters are exploding
        elif param.data.abs().max() > 1e+4:
            print(f'Parameter {name} is exploding (max abs value greater than 1e+10)')

        print(param.data.abs().max())


# Call the function
check_for_exploding_or_vanishing_params(model)

model.parameters()

# Suppose 'model' is your instance of STGCN
for name, param in model.named_parameters():
  if 'M' in name:  # If the parameter is M
    print(name)

    # Normalize the edge-importance matrix M
    normalized_M = (param.data - param.data.min()) / (param.data.max() - param.data.min())
    normalized_M_cpu = normalized_M.cpu().detach().numpy()
    # Plot the heatmap
    plt.imshow(normalized_M_cpu, cmap='coolwarm', vmin=0.90, vmax=1.1, interpolation='nearest')
    plt.colorbar()
    plt.title('Edge Importance Heatmap')
    plt.show()


for name, param in model.named_parameters():
  if 'M' in name:  # If the parameter is M
    print(name)

  param = torch.tensor(create_adjacency_matrix(joint_dimension)) + torch.eye(joint_dimension,joint_dimension)

  normalized_M = (param.data - param.data.min()) / (param.data.max() - param.data.min())
  normalized_M_cpu = normalized_M.cpu().detach().numpy()
  # Plot the heatmap
  plt.imshow(normalized_M_cpu, cmap='coolwarm', vmin=0.90, vmax=1.1, interpolation='nearest')
  plt.colorbar()
  plt.title('Adjacency Matrix')
  plt.show()

input_string = """Epoch 1/1000.. Train loss: 0.380.. Validation loss: 0.463
Epoch 2/1000.. Train loss: 0.283.. Validation loss: 0.397
Epoch 3/1000.. Train loss: 0.263.. Validation loss: 0.381
Epoch 4/1000.. Train loss: 0.255.. Validation loss: 0.354
Epoch 5/1000.. Train loss: 0.254.. Validation loss: 0.349
Epoch 6/1000.. Train loss: 0.255.. Validation loss: 0.342
Epoch 7/1000.. Train loss: 0.255.. Validation loss: 0.342
Epoch 8/1000.. Train loss: 0.254.. Validation loss: 0.335
Epoch 9/1000.. Train loss: 0.255.. Validation loss: 0.337
Epoch 10/1000.. Train loss: 0.256.. Validation loss: 0.330
Epoch 11/1000.. Train loss: 0.254.. Validation loss: 0.321
Epoch 12/1000.. Train loss: 0.252.. Validation loss: 0.330
Epoch 13/1000.. Train loss: 0.250.. Validation loss: 0.329
Epoch 14/1000.. Train loss: 0.253.. Validation loss: 0.331
Epoch 15/1000.. Train loss: 0.253.. Validation loss: 0.330
Epoch 16/1000.. Train loss: 0.251.. Validation loss: 0.324
Epoch 17/1000.. Train loss: 0.255.. Validation loss: 0.321
Epoch 18/1000.. Train loss: 0.255.. Validation loss: 0.319
Epoch 19/1000.. Train loss: 0.254.. Validation loss: 0.325
Epoch 20/1000.. Train loss: 0.252.. Validation loss: 0.320
Epoch 21/1000.. Train loss: 0.251.. Validation loss: 0.325
Epoch 22/1000.. Train loss: 0.253.. Validation loss: 0.324
Epoch 23/1000.. Train loss: 0.254.. Validation loss: 0.329
Epoch 24/1000.. Train loss: 0.248.. Validation loss: 0.323
Epoch 25/1000.. Train loss: 0.254.. Validation loss: 0.320
Epoch 26/1000.. Train loss: 0.254.. Validation loss: 0.321
Epoch 27/1000.. Train loss: 0.254.. Validation loss: 0.317
Epoch 28/1000.. Train loss: 0.253.. Validation loss: 0.313
Epoch 29/1000.. Train loss: 0.250.. Validation loss: 0.312
Epoch 30/1000.. Train loss: 0.249.. Validation loss: 0.314
Epoch 31/1000.. Train loss: 0.252.. Validation loss: 0.314
Epoch 32/1000.. Train loss: 0.253.. Validation loss: 0.315
Epoch 33/1000.. Train loss: 0.252.. Validation loss: 0.310
Epoch 34/1000.. Train loss: 0.254.. Validation loss: 0.313
Epoch 35/1000.. Train loss: 0.250.. Validation loss: 0.313
Epoch 36/1000.. Train loss: 0.250.. Validation loss: 0.311
Epoch 37/1000.. Train loss: 0.253.. Validation loss: 0.315
Epoch 38/1000.. Train loss: 0.251.. Validation loss: 0.314
Epoch 39/1000.. Train loss: 0.251.. Validation loss: 0.312
Epoch 40/1000.. Train loss: 0.255.. Validation loss: 0.314
Epoch 41/1000.. Train loss: 0.252.. Validation loss: 0.310
Epoch 42/1000.. Train loss: 0.254.. Validation loss: 0.315
Epoch 43/1000.. Train loss: 0.251.. Validation loss: 0.304
Epoch 44/1000.. Train loss: 0.252.. Validation loss: 0.308
Epoch 45/1000.. Train loss: 0.251.. Validation loss: 0.305
Epoch 46/1000.. Train loss: 0.250.. Validation loss: 0.304
Epoch 47/1000.. Train loss: 0.251.. Validation loss: 0.304
Epoch 48/1000.. Train loss: 0.252.. Validation loss: 0.303
Epoch 49/1000.. Train loss: 0.252.. Validation loss: 0.308
Epoch 50/1000.. Train loss: 0.252.. Validation loss: 0.305
Epoch 51/1000.. Train loss: 0.252.. Validation loss: 0.305
Epoch 52/1000.. Train loss: 0.254.. Validation loss: 0.300
Epoch 53/1000.. Train loss: 0.252.. Validation loss: 0.305
Epoch 54/1000.. Train loss: 0.251.. Validation loss: 0.301
Epoch 55/1000.. Train loss: 0.250.. Validation loss: 0.299
Epoch 56/1000.. Train loss: 0.250.. Validation loss: 0.303
Epoch 57/1000.. Train loss: 0.250.. Validation loss: 0.302
Epoch 58/1000.. Train loss: 0.250.. Validation loss: 0.302
Epoch 59/1000.. Train loss: 0.256.. Validation loss: 0.298
Epoch 60/1000.. Train loss: 0.256.. Validation loss: 0.303
Epoch 61/1000.. Train loss: 0.251.. Validation loss: 0.303
Epoch 62/1000.. Train loss: 0.250.. Validation loss: 0.301
Epoch 63/1000.. Train loss: 0.251.. Validation loss: 0.308
Epoch 64/1000.. Train loss: 0.253.. Validation loss: 0.306
Epoch 65/1000.. Train loss: 0.255.. Validation loss: 0.304
Epoch 66/1000.. Train loss: 0.251.. Validation loss: 0.308
Epoch 67/1000.. Train loss: 0.249.. Validation loss: 0.300
Epoch 68/1000.. Train loss: 0.250.. Validation loss: 0.305
Epoch 69/1000.. Train loss: 0.248.. Validation loss: 0.309
Epoch 70/1000.. Train loss: 0.249.. Validation loss: 0.303
Epoch 71/1000.. Train loss: 0.251.. Validation loss: 0.301
Epoch 72/1000.. Train loss: 0.254.. Validation loss: 0.300
Epoch 73/1000.. Train loss: 0.251.. Validation loss: 0.305
Epoch 74/1000.. Train loss: 0.252.. Validation loss: 0.303
Epoch 75/1000.. Train loss: 0.248.. Validation loss: 0.303
Epoch 76/1000.. Train loss: 0.252.. Validation loss: 0.304
Epoch 77/1000.. Train loss: 0.251.. Validation loss: 0.298
Epoch 78/1000.. Train loss: 0.251.. Validation loss: 0.301
Epoch 79/1000.. Train loss: 0.250.. Validation loss: 0.304
Epoch 80/1000.. Train loss: 0.252.. Validation loss: 0.302
Epoch 81/1000.. Train loss: 0.251.. Validation loss: 0.304
Epoch 82/1000.. Train loss: 0.253.. Validation loss: 0.302
Epoch 83/1000.. Train loss: 0.252.. Validation loss: 0.306
Epoch 84/1000.. Train loss: 0.251.. Validation loss: 0.300
Epoch 85/1000.. Train loss: 0.249.. Validation loss: 0.297
Epoch 86/1000.. Train loss: 0.252.. Validation loss: 0.306
Epoch 87/1000.. Train loss: 0.254.. Validation loss: 0.301
Epoch 88/1000.. Train loss: 0.252.. Validation loss: 0.304
Epoch 89/1000.. Train loss: 0.251.. Validation loss: 0.301
Epoch 90/1000.. Train loss: 0.251.. Validation loss: 0.297
Epoch 91/1000.. Train loss: 0.250.. Validation loss: 0.307
Epoch 92/1000.. Train loss: 0.250.. Validation loss: 0.297
Epoch 93/1000.. Train loss: 0.249.. Validation loss: 0.301
Epoch 94/1000.. Train loss: 0.250.. Validation loss: 0.301
Epoch 95/1000.. Train loss: 0.249.. Validation loss: 0.307
Epoch 96/1000.. Train loss: 0.249.. Validation loss: 0.305
Epoch 97/1000.. Train loss: 0.254.. Validation loss: 0.303
Epoch 98/1000.. Train loss: 0.249.. Validation loss: 0.300
Epoch 99/1000.. Train loss: 0.248.. Validation loss: 0.301
Epoch 100/1000.. Train loss: 0.248.. Validation loss: 0.297
Epoch 101/1000.. Train loss: 0.253.. Validation loss: 0.294
Epoch 102/1000.. Train loss: 0.251.. Validation loss: 0.299
Epoch 103/1000.. Train loss: 0.250.. Validation loss: 0.297
Epoch 104/1000.. Train loss: 0.252.. Validation loss: 0.292
Epoch 105/1000.. Train loss: 0.252.. Validation loss: 0.297
Epoch 106/1000.. Train loss: 0.252.. Validation loss: 0.301
Epoch 107/1000.. Train loss: 0.249.. Validation loss: 0.293
Epoch 108/1000.. Train loss: 0.252.. Validation loss: 0.299
Epoch 109/1000.. Train loss: 0.250.. Validation loss: 0.299
Epoch 110/1000.. Train loss: 0.250.. Validation loss: 0.300
Epoch 111/1000.. Train loss: 0.248.. Validation loss: 0.298
Epoch 112/1000.. Train loss: 0.253.. Validation loss: 0.302
Epoch 113/1000.. Train loss: 0.251.. Validation loss: 0.301
Epoch 114/1000.. Train loss: 0.251.. Validation loss: 0.296
Epoch 115/1000.. Train loss: 0.247.. Validation loss: 0.296
Epoch 116/1000.. Train loss: 0.251.. Validation loss: 0.297
Epoch 117/1000.. Train loss: 0.251.. Validation loss: 0.295
Epoch 118/1000.. Train loss: 0.249.. Validation loss: 0.297
Epoch 119/1000.. Train loss: 0.250.. Validation loss: 0.297
Epoch 120/1000.. Train loss: 0.249.. Validation loss: 0.296
Epoch 121/1000.. Train loss: 0.251.. Validation loss: 0.297
Epoch 122/1000.. Train loss: 0.248.. Validation loss: 0.298
Epoch 123/1000.. Train loss: 0.251.. Validation loss: 0.296
Epoch 124/1000.. Train loss: 0.248.. Validation loss: 0.293
Epoch 125/1000.. Train loss: 0.251.. Validation loss: 0.295
Epoch 126/1000.. Train loss: 0.252.. Validation loss: 0.291
Epoch 127/1000.. Train loss: 0.250.. Validation loss: 0.297
Epoch 128/1000.. Train loss: 0.250.. Validation loss: 0.298
Epoch 129/1000.. Train loss: 0.250.. Validation loss: 0.295
Epoch 130/1000.. Train loss: 0.247.. Validation loss: 0.294
Epoch 131/1000.. Train loss: 0.250.. Validation loss: 0.292
Epoch 132/1000.. Train loss: 0.246.. Validation loss: 0.289
Epoch 133/1000.. Train loss: 0.250.. Validation loss: 0.289
Epoch 134/1000.. Train loss: 0.251.. Validation loss: 0.290
Epoch 135/1000.. Train loss: 0.253.. Validation loss: 0.290
Epoch 136/1000.. Train loss: 0.250.. Validation loss: 0.294
Epoch 137/1000.. Train loss: 0.250.. Validation loss: 0.292
Epoch 138/1000.. Train loss: 0.248.. Validation loss: 0.293
Epoch 139/1000.. Train loss: 0.248.. Validation loss: 0.297
Epoch 140/1000.. Train loss: 0.249.. Validation loss: 0.298
Epoch 141/1000.. Train loss: 0.248.. Validation loss: 0.296
Epoch 142/1000.. Train loss: 0.251.. Validation loss: 0.292
Epoch 143/1000.. Train loss: 0.248.. Validation loss: 0.295
Epoch 144/1000.. Train loss: 0.250.. Validation loss: 0.291
Epoch 145/1000.. Train loss: 0.249.. Validation loss: 0.288
Epoch 146/1000.. Train loss: 0.248.. Validation loss: 0.293
Epoch 147/1000.. Train loss: 0.253.. Validation loss: 0.293
Epoch 148/1000.. Train loss: 0.248.. Validation loss: 0.291
Epoch 149/1000.. Train loss: 0.250.. Validation loss: 0.290
Epoch 150/1000.. Train loss: 0.248.. Validation loss: 0.288
Epoch 151/1000.. Train loss: 0.250.. Validation loss: 0.295
Epoch 152/1000.. Train loss: 0.246.. Validation loss: 0.290
Epoch 153/1000.. Train loss: 0.246.. Validation loss: 0.287
Epoch 154/1000.. Train loss: 0.247.. Validation loss: 0.296
Epoch 155/1000.. Train loss: 0.253.. Validation loss: 0.290
Epoch 156/1000.. Train loss: 0.248.. Validation loss: 0.288
Epoch 157/1000.. Train loss: 0.247.. Validation loss: 0.287
Epoch 158/1000.. Train loss: 0.250.. Validation loss: 0.288
Epoch 159/1000.. Train loss: 0.254.. Validation loss: 0.284
Epoch 160/1000.. Train loss: 0.246.. Validation loss: 0.288
Epoch 161/1000.. Train loss: 0.249.. Validation loss: 0.291
Epoch 162/1000.. Train loss: 0.250.. Validation loss: 0.286
Epoch 163/1000.. Train loss: 0.248.. Validation loss: 0.288
Epoch 164/1000.. Train loss: 0.250.. Validation loss: 0.285
Epoch 165/1000.. Train loss: 0.247.. Validation loss: 0.286
Epoch 166/1000.. Train loss: 0.253.. Validation loss: 0.286
Epoch 167/1000.. Train loss: 0.249.. Validation loss: 0.288
Epoch 168/1000.. Train loss: 0.252.. Validation loss: 0.285
Epoch 169/1000.. Train loss: 0.250.. Validation loss: 0.290
Epoch 170/1000.. Train loss: 0.249.. Validation loss: 0.288
Epoch 171/1000.. Train loss: 0.249.. Validation loss: 0.286
Epoch 172/1000.. Train loss: 0.248.. Validation loss: 0.287
Epoch 173/1000.. Train loss: 0.249.. Validation loss: 0.283
Epoch 174/1000.. Train loss: 0.248.. Validation loss: 0.284
Epoch 175/1000.. Train loss: 0.248.. Validation loss: 0.285
Epoch 176/1000.. Train loss: 0.250.. Validation loss: 0.283
Epoch 177/1000.. Train loss: 0.247.. Validation loss: 0.284
Epoch 178/1000.. Train loss: 0.248.. Validation loss: 0.286
Epoch 179/1000.. Train loss: 0.245.. Validation loss: 0.287
Epoch 180/1000.. Train loss: 0.250.. Validation loss: 0.286
Epoch 181/1000.. Train loss: 0.253.. Validation loss: 0.286
Epoch 182/1000.. Train loss: 0.247.. Validation loss: 0.287
Epoch 183/1000.. Train loss: 0.246.. Validation loss: 0.283
Epoch 184/1000.. Train loss: 0.250.. Validation loss: 0.279
Epoch 185/1000.. Train loss: 0.250.. Validation loss: 0.282
Epoch 186/1000.. Train loss: 0.249.. Validation loss: 0.282
Epoch 187/1000.. Train loss: 0.249.. Validation loss: 0.282
Epoch 188/1000.. Train loss: 0.246.. Validation loss: 0.280
Epoch 189/1000.. Train loss: 0.250.. Validation loss: 0.283
Epoch 190/1000.. Train loss: 0.252.. Validation loss: 0.283
Epoch 191/1000.. Train loss: 0.248.. Validation loss: 0.279
Epoch 192/1000.. Train loss: 0.251.. Validation loss: 0.284
Epoch 193/1000.. Train loss: 0.247.. Validation loss: 0.286
Epoch 194/1000.. Train loss: 0.246.. Validation loss: 0.279
Epoch 195/1000.. Train loss: 0.250.. Validation loss: 0.280
Epoch 196/1000.. Train loss: 0.249.. Validation loss: 0.279
Epoch 197/1000.. Train loss: 0.249.. Validation loss: 0.279
Epoch 198/1000.. Train loss: 0.247.. Validation loss: 0.279
Epoch 199/1000.. Train loss: 0.247.. Validation loss: 0.279
Epoch 200/1000.. Train loss: 0.249.. Validation loss: 0.280
Epoch 201/1000.. Train loss: 0.249.. Validation loss: 0.279
Epoch 202/1000.. Train loss: 0.246.. Validation loss: 0.277
Epoch 203/1000.. Train loss: 0.245.. Validation loss: 0.277
Epoch 204/1000.. Train loss: 0.249.. Validation loss: 0.276
Epoch 205/1000.. Train loss: 0.249.. Validation loss: 0.274
Epoch 206/1000.. Train loss: 0.252.. Validation loss: 0.275
Epoch 207/1000.. Train loss: 0.253.. Validation loss: 0.270
Epoch 208/1000.. Train loss: 0.246.. Validation loss: 0.274
Epoch 209/1000.. Train loss: 0.249.. Validation loss: 0.272
Epoch 210/1000.. Train loss: 0.249.. Validation loss: 0.270
Epoch 211/1000.. Train loss: 0.250.. Validation loss: 0.271
Epoch 212/1000.. Train loss: 0.247.. Validation loss: 0.273
Epoch 213/1000.. Train loss: 0.252.. Validation loss: 0.271
Epoch 214/1000.. Train loss: 0.251.. Validation loss: 0.274
Epoch 215/1000.. Train loss: 0.248.. Validation loss: 0.274
Epoch 216/1000.. Train loss: 0.246.. Validation loss: 0.268
Epoch 217/1000.. Train loss: 0.247.. Validation loss: 0.272
Epoch 218/1000.. Train loss: 0.246.. Validation loss: 0.275
Epoch 219/1000.. Train loss: 0.250.. Validation loss: 0.272
Epoch 220/1000.. Train loss: 0.255.. Validation loss: 0.274
Epoch 221/1000.. Train loss: 0.250.. Validation loss: 0.272
Epoch 222/1000.. Train loss: 0.248.. Validation loss: 0.275
Epoch 223/1000.. Train loss: 0.250.. Validation loss: 0.277
Epoch 224/1000.. Train loss: 0.250.. Validation loss: 0.274
Epoch 225/1000.. Train loss: 0.250.. Validation loss: 0.273
Epoch 226/1000.. Train loss: 0.250.. Validation loss: 0.275
Epoch 227/1000.. Train loss: 0.248.. Validation loss: 0.274
Epoch 228/1000.. Train loss: 0.245.. Validation loss: 0.275
Epoch 229/1000.. Train loss: 0.246.. Validation loss: 0.274
Epoch 230/1000.. Train loss: 0.249.. Validation loss: 0.271
Epoch 231/1000.. Train loss: 0.249.. Validation loss: 0.269
Epoch 232/1000.. Train loss: 0.244.. Validation loss: 0.270
Epoch 233/1000.. Train loss: 0.249.. Validation loss: 0.272
Epoch 234/1000.. Train loss: 0.253.. Validation loss: 0.276
Epoch 235/1000.. Train loss: 0.251.. Validation loss: 0.272
Epoch 236/1000.. Train loss: 0.253.. Validation loss: 0.270
Epoch 237/1000.. Train loss: 0.246.. Validation loss: 0.274
Epoch 238/1000.. Train loss: 0.248.. Validation loss: 0.268
Epoch 239/1000.. Train loss: 0.245.. Validation loss: 0.270
Epoch 240/1000.. Train loss: 0.247.. Validation loss: 0.271
Epoch 241/1000.. Train loss: 0.248.. Validation loss: 0.272
Epoch 242/1000.. Train loss: 0.250.. Validation loss: 0.274
Epoch 243/1000.. Train loss: 0.246.. Validation loss: 0.271
Epoch 244/1000.. Train loss: 0.251.. Validation loss: 0.272
Epoch 245/1000.. Train loss: 0.244.. Validation loss: 0.270
Epoch 246/1000.. Train loss: 0.247.. Validation loss: 0.266
Epoch 247/1000.. Train loss: 0.250.. Validation loss: 0.267
Epoch 248/1000.. Train loss: 0.248.. Validation loss: 0.269
Epoch 249/1000.. Train loss: 0.246.. Validation loss: 0.268
Epoch 250/1000.. Train loss: 0.247.. Validation loss: 0.269
Epoch 251/1000.. Train loss: 0.245.. Validation loss: 0.271
Epoch 252/1000.. Train loss: 0.249.. Validation loss: 0.272
Epoch 253/1000.. Train loss: 0.247.. Validation loss: 0.274
Epoch 254/1000.. Train loss: 0.246.. Validation loss: 0.270
Epoch 255/1000.. Train loss: 0.245.. Validation loss: 0.270
Epoch 256/1000.. Train loss: 0.248.. Validation loss: 0.269
Epoch 257/1000.. Train loss: 0.250.. Validation loss: 0.265
Epoch 258/1000.. Train loss: 0.247.. Validation loss: 0.269
Epoch 259/1000.. Train loss: 0.251.. Validation loss: 0.267
Epoch 260/1000.. Train loss: 0.251.. Validation loss: 0.271
Epoch 261/1000.. Train loss: 0.250.. Validation loss: 0.270
Epoch 262/1000.. Train loss: 0.243.. Validation loss: 0.269
Epoch 263/1000.. Train loss: 0.248.. Validation loss: 0.268
Epoch 264/1000.. Train loss: 0.241.. Validation loss: 0.265
Epoch 265/1000.. Train loss: 0.253.. Validation loss: 0.266
Epoch 266/1000.. Train loss: 0.249.. Validation loss: 0.266
Epoch 267/1000.. Train loss: 0.250.. Validation loss: 0.267
Epoch 268/1000.. Train loss: 0.245.. Validation loss: 0.265
Epoch 269/1000.. Train loss: 0.249.. Validation loss: 0.266
Epoch 270/1000.. Train loss: 0.252.. Validation loss: 0.262
Epoch 271/1000.. Train loss: 0.243.. Validation loss: 0.265
Epoch 272/1000.. Train loss: 0.251.. Validation loss: 0.265
Epoch 273/1000.. Train loss: 0.246.. Validation loss: 0.264
Epoch 274/1000.. Train loss: 0.243.. Validation loss: 0.265
Epoch 275/1000.. Train loss: 0.240.. Validation loss: 0.264
Epoch 276/1000.. Train loss: 0.245.. Validation loss: 0.262
Epoch 277/1000.. Train loss: 0.246.. Validation loss: 0.266
Epoch 278/1000.. Train loss: 0.252.. Validation loss: 0.260
Epoch 279/1000.. Train loss: 0.248.. Validation loss: 0.263
Epoch 280/1000.. Train loss: 0.242.. Validation loss: 0.262
Epoch 281/1000.. Train loss: 0.249.. Validation loss: 0.264
Epoch 282/1000.. Train loss: 0.246.. Validation loss: 0.262
Epoch 283/1000.. Train loss: 0.247.. Validation loss: 0.263
Epoch 284/1000.. Train loss: 0.246.. Validation loss: 0.263
Epoch 285/1000.. Train loss: 0.239.. Validation loss: 0.262
Epoch 286/1000.. Train loss: 0.242.. Validation loss: 0.263
Epoch 287/1000.. Train loss: 0.248.. Validation loss: 0.260
Epoch 288/1000.. Train loss: 0.243.. Validation loss: 0.259
Epoch 289/1000.. Train loss: 0.248.. Validation loss: 0.261
Epoch 290/1000.. Train loss: 0.243.. Validation loss: 0.261
Epoch 291/1000.. Train loss: 0.243.. Validation loss: 0.263
Epoch 292/1000.. Train loss: 0.241.. Validation loss: 0.262
Epoch 293/1000.. Train loss: 0.249.. Validation loss: 0.261
Epoch 294/1000.. Train loss: 0.242.. Validation loss: 0.256
Epoch 295/1000.. Train loss: 0.244.. Validation loss: 0.261
Epoch 296/1000.. Train loss: 0.244.. Validation loss: 0.258
Epoch 297/1000.. Train loss: 0.245.. Validation loss: 0.257
Epoch 298/1000.. Train loss: 0.248.. Validation loss: 0.258
Epoch 299/1000.. Train loss: 0.250.. Validation loss: 0.259
Epoch 300/1000.. Train loss: 0.247.. Validation loss: 0.257
Epoch 301/1000.. Train loss: 0.248.. Validation loss: 0.258
Epoch 302/1000.. Train loss: 0.244.. Validation loss: 0.256
Epoch 303/1000.. Train loss: 0.244.. Validation loss: 0.260
Epoch 304/1000.. Train loss: 0.245.. Validation loss: 0.258
Epoch 305/1000.. Train loss: 0.246.. Validation loss: 0.255
Epoch 306/1000.. Train loss: 0.247.. Validation loss: 0.257
Epoch 307/1000.. Train loss: 0.240.. Validation loss: 0.259
Epoch 308/1000.. Train loss: 0.243.. Validation loss: 0.258
Epoch 309/1000.. Train loss: 0.245.. Validation loss: 0.254
Epoch 310/1000.. Train loss: 0.248.. Validation loss: 0.254
Epoch 311/1000.. Train loss: 0.243.. Validation loss: 0.255
Epoch 312/1000.. Train loss: 0.249.. Validation loss: 0.257
Epoch 313/1000.. Train loss: 0.240.. Validation loss: 0.256
Epoch 314/1000.. Train loss: 0.252.. Validation loss: 0.253
Epoch 315/1000.. Train loss: 0.245.. Validation loss: 0.257
Epoch 316/1000.. Train loss: 0.243.. Validation loss: 0.256
Epoch 317/1000.. Train loss: 0.245.. Validation loss: 0.254
Epoch 318/1000.. Train loss: 0.240.. Validation loss: 0.257
Epoch 319/1000.. Train loss: 0.241.. Validation loss: 0.252
Epoch 320/1000.. Train loss: 0.238.. Validation loss: 0.251
Epoch 321/1000.. Train loss: 0.243.. Validation loss: 0.255
Epoch 322/1000.. Train loss: 0.241.. Validation loss: 0.251
Epoch 323/1000.. Train loss: 0.248.. Validation loss: 0.253
Epoch 324/1000.. Train loss: 0.243.. Validation loss: 0.252
Epoch 325/1000.. Train loss: 0.241.. Validation loss: 0.252
Epoch 326/1000.. Train loss: 0.246.. Validation loss: 0.253
Epoch 327/1000.. Train loss: 0.238.. Validation loss: 0.250
Epoch 328/1000.. Train loss: 0.242.. Validation loss: 0.254
Epoch 329/1000.. Train loss: 0.238.. Validation loss: 0.252
Epoch 330/1000.. Train loss: 0.249.. Validation loss: 0.253
Epoch 331/1000.. Train loss: 0.243.. Validation loss: 0.249
Epoch 332/1000.. Train loss: 0.247.. Validation loss: 0.250
Epoch 333/1000.. Train loss: 0.244.. Validation loss: 0.253
Epoch 334/1000.. Train loss: 0.248.. Validation loss: 0.253
Epoch 335/1000.. Train loss: 0.245.. Validation loss: 0.249
Epoch 336/1000.. Train loss: 0.245.. Validation loss: 0.249
Epoch 337/1000.. Train loss: 0.244.. Validation loss: 0.249
Epoch 338/1000.. Train loss: 0.240.. Validation loss: 0.248
Epoch 339/1000.. Train loss: 0.243.. Validation loss: 0.250
Epoch 340/1000.. Train loss: 0.238.. Validation loss: 0.249
Epoch 341/1000.. Train loss: 0.238.. Validation loss: 0.249
Epoch 342/1000.. Train loss: 0.246.. Validation loss: 0.251
Epoch 343/1000.. Train loss: 0.243.. Validation loss: 0.248
Epoch 344/1000.. Train loss: 0.248.. Validation loss: 0.249
Epoch 345/1000.. Train loss: 0.240.. Validation loss: 0.247
Epoch 346/1000.. Train loss: 0.244.. Validation loss: 0.249
Epoch 347/1000.. Train loss: 0.241.. Validation loss: 0.249
Epoch 348/1000.. Train loss: 0.242.. Validation loss: 0.248
Epoch 349/1000.. Train loss: 0.246.. Validation loss: 0.246
Epoch 350/1000.. Train loss: 0.240.. Validation loss: 0.250
Epoch 351/1000.. Train loss: 0.235.. Validation loss: 0.250
Epoch 352/1000.. Train loss: 0.242.. Validation loss: 0.247
Epoch 353/1000.. Train loss: 0.248.. Validation loss: 0.248
Epoch 354/1000.. Train loss: 0.237.. Validation loss: 0.251
Epoch 355/1000.. Train loss: 0.238.. Validation loss: 0.251
Epoch 356/1000.. Train loss: 0.231.. Validation loss: 0.246
Epoch 357/1000.. Train loss: 0.244.. Validation loss: 0.248
Epoch 358/1000.. Train loss: 0.241.. Validation loss: 0.246
Epoch 359/1000.. Train loss: 0.241.. Validation loss: 0.248
Epoch 360/1000.. Train loss: 0.240.. Validation loss: 0.246
Epoch 361/1000.. Train loss: 0.239.. Validation loss: 0.246
Epoch 362/1000.. Train loss: 0.236.. Validation loss: 0.245
Epoch 363/1000.. Train loss: 0.240.. Validation loss: 0.247
Epoch 364/1000.. Train loss: 0.242.. Validation loss: 0.246
Epoch 365/1000.. Train loss: 0.241.. Validation loss: 0.247
Epoch 366/1000.. Train loss: 0.244.. Validation loss: 0.245
Epoch 367/1000.. Train loss: 0.238.. Validation loss: 0.245
Epoch 368/1000.. Train loss: 0.240.. Validation loss: 0.243
Epoch 369/1000.. Train loss: 0.241.. Validation loss: 0.244
Epoch 370/1000.. Train loss: 0.237.. Validation loss: 0.243
Epoch 371/1000.. Train loss: 0.237.. Validation loss: 0.246
Epoch 372/1000.. Train loss: 0.240.. Validation loss: 0.244
Epoch 373/1000.. Train loss: 0.237.. Validation loss: 0.243
Epoch 374/1000.. Train loss: 0.237.. Validation loss: 0.243
Epoch 375/1000.. Train loss: 0.246.. Validation loss: 0.244
Epoch 376/1000.. Train loss: 0.240.. Validation loss: 0.244
Epoch 377/1000.. Train loss: 0.236.. Validation loss: 0.245
Epoch 378/1000.. Train loss: 0.233.. Validation loss: 0.246
Epoch 379/1000.. Train loss: 0.238.. Validation loss: 0.241
Epoch 380/1000.. Train loss: 0.235.. Validation loss: 0.244
Epoch 381/1000.. Train loss: 0.240.. Validation loss: 0.243
Epoch 382/1000.. Train loss: 0.236.. Validation loss: 0.242
Epoch 383/1000.. Train loss: 0.246.. Validation loss: 0.242
Epoch 384/1000.. Train loss: 0.242.. Validation loss: 0.244
Epoch 385/1000.. Train loss: 0.241.. Validation loss: 0.244
Epoch 386/1000.. Train loss: 0.248.. Validation loss: 0.245
Epoch 387/1000.. Train loss: 0.239.. Validation loss: 0.242
Epoch 388/1000.. Train loss: 0.237.. Validation loss: 0.243
Epoch 389/1000.. Train loss: 0.249.. Validation loss: 0.243
Epoch 390/1000.. Train loss: 0.237.. Validation loss: 0.240
Epoch 391/1000.. Train loss: 0.236.. Validation loss: 0.242
Epoch 392/1000.. Train loss: 0.243.. Validation loss: 0.240
Epoch 393/1000.. Train loss: 0.241.. Validation loss: 0.243
Epoch 394/1000.. Train loss: 0.244.. Validation loss: 0.241
Epoch 395/1000.. Train loss: 0.232.. Validation loss: 0.240
Epoch 396/1000.. Train loss: 0.245.. Validation loss: 0.241
Epoch 397/1000.. Train loss: 0.243.. Validation loss: 0.241
Epoch 398/1000.. Train loss: 0.237.. Validation loss: 0.241
Epoch 399/1000.. Train loss: 0.244.. Validation loss: 0.241
Epoch 400/1000.. Train loss: 0.243.. Validation loss: 0.239
Epoch 401/1000.. Train loss: 0.242.. Validation loss: 0.241
Epoch 402/1000.. Train loss: 0.240.. Validation loss: 0.241
Epoch 403/1000.. Train loss: 0.238.. Validation loss: 0.241
Epoch 404/1000.. Train loss: 0.250.. Validation loss: 0.241
Epoch 405/1000.. Train loss: 0.241.. Validation loss: 0.242
Epoch 406/1000.. Train loss: 0.236.. Validation loss: 0.242
Epoch 407/1000.. Train loss: 0.236.. Validation loss: 0.242
Epoch 408/1000.. Train loss: 0.239.. Validation loss: 0.243
Epoch 409/1000.. Train loss: 0.242.. Validation loss: 0.240
Epoch 410/1000.. Train loss: 0.239.. Validation loss: 0.241
Epoch 411/1000.. Train loss: 0.249.. Validation loss: 0.241
Epoch 412/1000.. Train loss: 0.244.. Validation loss: 0.240
Epoch 413/1000.. Train loss: 0.237.. Validation loss: 0.240
Epoch 414/1000.. Train loss: 0.239.. Validation loss: 0.239
Epoch 415/1000.. Train loss: 0.236.. Validation loss: 0.239
Epoch 416/1000.. Train loss: 0.236.. Validation loss: 0.241
Epoch 417/1000.. Train loss: 0.239.. Validation loss: 0.240
Epoch 418/1000.. Train loss: 0.236.. Validation loss: 0.240
Epoch 419/1000.. Train loss: 0.234.. Validation loss: 0.241
Epoch 420/1000.. Train loss: 0.245.. Validation loss: 0.240
Epoch 421/1000.. Train loss: 0.236.. Validation loss: 0.238
Epoch 422/1000.. Train loss: 0.235.. Validation loss: 0.239
Epoch 423/1000.. Train loss: 0.243.. Validation loss: 0.240
Epoch 424/1000.. Train loss: 0.243.. Validation loss: 0.240
Epoch 425/1000.. Train loss: 0.247.. Validation loss: 0.241
Epoch 426/1000.. Train loss: 0.235.. Validation loss: 0.241
Epoch 427/1000.. Train loss: 0.240.. Validation loss: 0.241
Epoch 428/1000.. Train loss: 0.240.. Validation loss: 0.240
Epoch 429/1000.. Train loss: 0.241.. Validation loss: 0.240
Epoch 430/1000.. Train loss: 0.244.. Validation loss: 0.239
Epoch 431/1000.. Train loss: 0.236.. Validation loss: 0.242
Epoch 432/1000.. Train loss: 0.244.. Validation loss: 0.240
Epoch 433/1000.. Train loss: 0.236.. Validation loss: 0.239
Epoch 434/1000.. Train loss: 0.231.. Validation loss: 0.240
Epoch 435/1000.. Train loss: 0.240.. Validation loss: 0.240
Epoch 436/1000.. Train loss: 0.238.. Validation loss: 0.240
Epoch 437/1000.. Train loss: 0.233.. Validation loss: 0.240
Epoch 438/1000.. Train loss: 0.236.. Validation loss: 0.240
Epoch 439/1000.. Train loss: 0.245.. Validation loss: 0.240
Epoch 440/1000.. Train loss: 0.238.. Validation loss: 0.241
Epoch 441/1000.. Train loss: 0.239.. Validation loss: 0.240
Epoch 442/1000.. Train loss: 0.245.. Validation loss: 0.239
Epoch 443/1000.. Train loss: 0.228.. Validation loss: 0.239
Epoch 444/1000.. Train loss: 0.234.. Validation loss: 0.240
Epoch 445/1000.. Train loss: 0.240.. Validation loss: 0.239
Epoch 446/1000.. Train loss: 0.249.. Validation loss: 0.239
Epoch 447/1000.. Train loss: 0.230.. Validation loss: 0.238
Epoch 448/1000.. Train loss: 0.238.. Validation loss: 0.239
Epoch 449/1000.. Train loss: 0.237.. Validation loss: 0.239
Epoch 450/1000.. Train loss: 0.235.. Validation loss: 0.238
Epoch 451/1000.. Train loss: 0.243.. Validation loss: 0.239
Epoch 452/1000.. Train loss: 0.240.. Validation loss: 0.238
Epoch 453/1000.. Train loss: 0.238.. Validation loss: 0.239
Epoch 454/1000.. Train loss: 0.240.. Validation loss: 0.238
Epoch 455/1000.. Train loss: 0.228.. Validation loss: 0.238
Epoch 456/1000.. Train loss: 0.248.. Validation loss: 0.238
Epoch 457/1000.. Train loss: 0.241.. Validation loss: 0.239
Epoch 458/1000.. Train loss: 0.234.. Validation loss: 0.238
Epoch 459/1000.. Train loss: 0.229.. Validation loss: 0.238
Epoch 460/1000.. Train loss: 0.241.. Validation loss: 0.238
Epoch 461/1000.. Train loss: 0.233.. Validation loss: 0.239
Epoch 462/1000.. Train loss: 0.228.. Validation loss: 0.237
Epoch 463/1000.. Train loss: 0.242.. Validation loss: 0.237
Epoch 464/1000.. Train loss: 0.231.. Validation loss: 0.238
Epoch 465/1000.. Train loss: 0.235.. Validation loss: 0.237
Epoch 466/1000.. Train loss: 0.246.. Validation loss: 0.237
Epoch 467/1000.. Train loss: 0.238.. Validation loss: 0.236
Epoch 468/1000.. Train loss: 0.231.. Validation loss: 0.237
Epoch 469/1000.. Train loss: 0.229.. Validation loss: 0.237
Epoch 470/1000.. Train loss: 0.232.. Validation loss: 0.237
Epoch 471/1000.. Train loss: 0.238.. Validation loss: 0.237
Epoch 472/1000.. Train loss: 0.226.. Validation loss: 0.236
Epoch 473/1000.. Train loss: 0.234.. Validation loss: 0.236
Epoch 474/1000.. Train loss: 0.237.. Validation loss: 0.235
Epoch 475/1000.. Train loss: 0.243.. Validation loss: 0.236
Epoch 476/1000.. Train loss: 0.240.. Validation loss: 0.237
Epoch 477/1000.. Train loss: 0.246.. Validation loss: 0.237
Epoch 478/1000.. Train loss: 0.241.. Validation loss: 0.236
Epoch 479/1000.. Train loss: 0.234.. Validation loss: 0.236
Epoch 480/1000.. Train loss: 0.237.. Validation loss: 0.237
Epoch 481/1000.. Train loss: 0.239.. Validation loss: 0.237
Epoch 482/1000.. Train loss: 0.237.. Validation loss: 0.236
Epoch 483/1000.. Train loss: 0.230.. Validation loss: 0.237
Epoch 484/1000.. Train loss: 0.238.. Validation loss: 0.237
Epoch 485/1000.. Train loss: 0.240.. Validation loss: 0.237
Epoch 486/1000.. Train loss: 0.238.. Validation loss: 0.237
Epoch 487/1000.. Train loss: 0.247.. Validation loss: 0.237
Epoch 488/1000.. Train loss: 0.240.. Validation loss: 0.237
Epoch 489/1000.. Train loss: 0.238.. Validation loss: 0.237
Epoch 490/1000.. Train loss: 0.241.. Validation loss: 0.237
Epoch 491/1000.. Train loss: 0.232.. Validation loss: 0.236
Epoch 492/1000.. Train loss: 0.234.. Validation loss: 0.237
Epoch 493/1000.. Train loss: 0.231.. Validation loss: 0.236
Epoch 494/1000.. Train loss: 0.239.. Validation loss: 0.237
Epoch 495/1000.. Train loss: 0.244.. Validation loss: 0.237
Epoch 496/1000.. Train loss: 0.243.. Validation loss: 0.236
Epoch 497/1000.. Train loss: 0.234.. Validation loss: 0.237
Epoch 498/1000.. Train loss: 0.237.. Validation loss: 0.236
Epoch 499/1000.. Train loss: 0.238.. Validation loss: 0.236
Epoch 500/1000.. Train loss: 0.242.. Validation loss: 0.236
Epoch 501/1000.. Train loss: 0.237.. Validation loss: 0.236
Epoch 502/1000.. Train loss: 0.239.. Validation loss: 0.236
Epoch 503/1000.. Train loss: 0.231.. Validation loss: 0.236
Epoch 504/1000.. Train loss: 0.231.. Validation loss: 0.235
Epoch 505/1000.. Train loss: 0.241.. Validation loss: 0.235
Epoch 506/1000.. Train loss: 0.232.. Validation loss: 0.235
Epoch 507/1000.. Train loss: 0.238.. Validation loss: 0.235
Epoch 508/1000.. Train loss: 0.237.. Validation loss: 0.235
Epoch 509/1000.. Train loss: 0.232.. Validation loss: 0.235
Epoch 510/1000.. Train loss: 0.236.. Validation loss: 0.235
Epoch 511/1000.. Train loss: 0.223.. Validation loss: 0.234
Epoch 512/1000.. Train loss: 0.242.. Validation loss: 0.235
Epoch 513/1000.. Train loss: 0.230.. Validation loss: 0.235
Epoch 514/1000.. Train loss: 0.238.. Validation loss: 0.235
Epoch 515/1000.. Train loss: 0.246.. Validation loss: 0.235
Epoch 516/1000.. Train loss: 0.236.. Validation loss: 0.235
Epoch 517/1000.. Train loss: 0.234.. Validation loss: 0.235
Epoch 518/1000.. Train loss: 0.238.. Validation loss: 0.235
Epoch 519/1000.. Train loss: 0.233.. Validation loss: 0.235
Epoch 520/1000.. Train loss: 0.237.. Validation loss: 0.235
Epoch 521/1000.. Train loss: 0.234.. Validation loss: 0.236
Epoch 522/1000.. Train loss: 0.237.. Validation loss: 0.235
Epoch 523/1000.. Train loss: 0.240.. Validation loss: 0.235
Epoch 524/1000.. Train loss: 0.244.. Validation loss: 0.235
Epoch 525/1000.. Train loss: 0.238.. Validation loss: 0.235
Epoch 526/1000.. Train loss: 0.237.. Validation loss: 0.236
Epoch 527/1000.. Train loss: 0.236.. Validation loss: 0.235
Epoch 528/1000.. Train loss: 0.230.. Validation loss: 0.235
Epoch 529/1000.. Train loss: 0.230.. Validation loss: 0.235
Epoch 530/1000.. Train loss: 0.235.. Validation loss: 0.235
Epoch 531/1000.. Train loss: 0.240.. Validation loss: 0.236
Epoch 532/1000.. Train loss: 0.237.. Validation loss: 0.236
Epoch 533/1000.. Train loss: 0.234.. Validation loss: 0.236
Epoch 534/1000.. Train loss: 0.225.. Validation loss: 0.235
Epoch 535/1000.. Train loss: 0.241.. Validation loss: 0.235
Epoch 536/1000.. Train loss: 0.222.. Validation loss: 0.239
Epoch 537/1000.. Train loss: 0.234.. Validation loss: 0.235
Epoch 538/1000.. Train loss: 0.229.. Validation loss: 0.235
Epoch 539/1000.. Train loss: 0.237.. Validation loss: 0.235
Epoch 540/1000.. Train loss: 0.227.. Validation loss: 0.236
Epoch 541/1000.. Train loss: 0.237.. Validation loss: 0.235
Epoch 542/1000.. Train loss: 0.240.. Validation loss: 0.234
Epoch 543/1000.. Train loss: 0.240.. Validation loss: 0.237
Epoch 544/1000.. Train loss: 0.228.. Validation loss: 0.234
Epoch 545/1000.. Train loss: 0.247.. Validation loss: 0.235
Epoch 546/1000.. Train loss: 0.239.. Validation loss: 0.234
Epoch 547/1000.. Train loss: 0.226.. Validation loss: 0.235
Epoch 548/1000.. Train loss: 0.240.. Validation loss: 0.235
Epoch 549/1000.. Train loss: 0.235.. Validation loss: 0.234
Epoch 550/1000.. Train loss: 0.247.. Validation loss: 0.233
Epoch 551/1000.. Train loss: 0.238.. Validation loss: 0.235
Epoch 552/1000.. Train loss: 0.231.. Validation loss: 0.233
Epoch 553/1000.. Train loss: 0.232.. Validation loss: 0.235
Epoch 554/1000.. Train loss: 0.238.. Validation loss: 0.234
Epoch 555/1000.. Train loss: 0.242.. Validation loss: 0.233
Epoch 556/1000.. Train loss: 0.235.. Validation loss: 0.234
Epoch 557/1000.. Train loss: 0.231.. Validation loss: 0.233
Epoch 558/1000.. Train loss: 0.234.. Validation loss: 0.233
Epoch 559/1000.. Train loss: 0.226.. Validation loss: 0.234
Epoch 560/1000.. Train loss: 0.250.. Validation loss: 0.233
Epoch 561/1000.. Train loss: 0.242.. Validation loss: 0.234
Epoch 562/1000.. Train loss: 0.235.. Validation loss: 0.234
Epoch 563/1000.. Train loss: 0.237.. Validation loss: 0.233
Epoch 564/1000.. Train loss: 0.246.. Validation loss: 0.233
Epoch 565/1000.. Train loss: 0.234.. Validation loss: 0.234
Epoch 566/1000.. Train loss: 0.231.. Validation loss: 0.233
Epoch 567/1000.. Train loss: 0.233.. Validation loss: 0.233
Epoch 568/1000.. Train loss: 0.235.. Validation loss: 0.233
Epoch 569/1000.. Train loss: 0.222.. Validation loss: 0.233
Epoch 570/1000.. Train loss: 0.241.. Validation loss: 0.232
Epoch 571/1000.. Train loss: 0.226.. Validation loss: 0.233
Epoch 572/1000.. Train loss: 0.232.. Validation loss: 0.234
Epoch 573/1000.. Train loss: 0.237.. Validation loss: 0.232
Epoch 574/1000.. Train loss: 0.233.. Validation loss: 0.233
Epoch 575/1000.. Train loss: 0.238.. Validation loss: 0.232
Epoch 576/1000.. Train loss: 0.238.. Validation loss: 0.233
Epoch 577/1000.. Train loss: 0.227.. Validation loss: 0.232
Epoch 578/1000.. Train loss: 0.239.. Validation loss: 0.233
Epoch 579/1000.. Train loss: 0.245.. Validation loss: 0.232
Epoch 580/1000.. Train loss: 0.224.. Validation loss: 0.232
Epoch 581/1000.. Train loss: 0.233.. Validation loss: 0.233
Epoch 582/1000.. Train loss: 0.238.. Validation loss: 0.233
Epoch 583/1000.. Train loss: 0.238.. Validation loss: 0.233
Epoch 584/1000.. Train loss: 0.228.. Validation loss: 0.232
Epoch 585/1000.. Train loss: 0.234.. Validation loss: 0.233
Epoch 586/1000.. Train loss: 0.224.. Validation loss: 0.232
Epoch 587/1000.. Train loss: 0.232.. Validation loss: 0.233
Epoch 588/1000.. Train loss: 0.243.. Validation loss: 0.232
Epoch 589/1000.. Train loss: 0.240.. Validation loss: 0.232
Epoch 590/1000.. Train loss: 0.236.. Validation loss: 0.232
Epoch 591/1000.. Train loss: 0.231.. Validation loss: 0.233
Epoch 592/1000.. Train loss: 0.239.. Validation loss: 0.233
Epoch 593/1000.. Train loss: 0.229.. Validation loss: 0.234
Epoch 594/1000.. Train loss: 0.233.. Validation loss: 0.232
Epoch 595/1000.. Train loss: 0.231.. Validation loss: 0.231
Epoch 596/1000.. Train loss: 0.241.. Validation loss: 0.232
Epoch 597/1000.. Train loss: 0.244.. Validation loss: 0.232
Epoch 598/1000.. Train loss: 0.232.. Validation loss: 0.236
Epoch 599/1000.. Train loss: 0.235.. Validation loss: 0.233
Epoch 600/1000.. Train loss: 0.224.. Validation loss: 0.232
Epoch 601/1000.. Train loss: 0.240.. Validation loss: 0.232
Epoch 602/1000.. Train loss: 0.238.. Validation loss: 0.232
Epoch 603/1000.. Train loss: 0.238.. Validation loss: 0.233
Epoch 604/1000.. Train loss: 0.227.. Validation loss: 0.232
Epoch 605/1000.. Train loss: 0.237.. Validation loss: 0.232
Epoch 606/1000.. Train loss: 0.232.. Validation loss: 0.232
Epoch 607/1000.. Train loss: 0.222.. Validation loss: 0.232
Epoch 608/1000.. Train loss: 0.226.. Validation loss: 0.233
Epoch 609/1000.. Train loss: 0.229.. Validation loss: 0.232
Epoch 610/1000.. Train loss: 0.232.. Validation loss: 0.232
Epoch 611/1000.. Train loss: 0.237.. Validation loss: 0.234
Epoch 612/1000.. Train loss: 0.240.. Validation loss: 0.232
Epoch 613/1000.. Train loss: 0.236.. Validation loss: 0.234
Epoch 614/1000.. Train loss: 0.228.. Validation loss: 0.233
Epoch 615/1000.. Train loss: 0.232.. Validation loss: 0.234
Epoch 616/1000.. Train loss: 0.245.. Validation loss: 0.233
Epoch 617/1000.. Train loss: 0.230.. Validation loss: 0.234
Epoch 618/1000.. Train loss: 0.236.. Validation loss: 0.232
Epoch 619/1000.. Train loss: 0.241.. Validation loss: 0.232
Epoch 620/1000.. Train loss: 0.229.. Validation loss: 0.232
Epoch 621/1000.. Train loss: 0.242.. Validation loss: 0.233
Epoch 622/1000.. Train loss: 0.234.. Validation loss: 0.233
Epoch 623/1000.. Train loss: 0.229.. Validation loss: 0.232
Epoch 624/1000.. Train loss: 0.231.. Validation loss: 0.232
Epoch 625/1000.. Train loss: 0.227.. Validation loss: 0.232
Epoch 626/1000.. Train loss: 0.228.. Validation loss: 0.234
Epoch 627/1000.. Train loss: 0.236.. Validation loss: 0.232
Epoch 628/1000.. Train loss: 0.224.. Validation loss: 0.231
Epoch 629/1000.. Train loss: 0.229.. Validation loss: 0.233
Epoch 630/1000.. Train loss: 0.231.. Validation loss: 0.233
Epoch 631/1000.. Train loss: 0.231.. Validation loss: 0.235
Epoch 632/1000.. Train loss: 0.231.. Validation loss: 0.234
Epoch 633/1000.. Train loss: 0.241.. Validation loss: 0.235
Epoch 634/1000.. Train loss: 0.227.. Validation loss: 0.234
Epoch 635/1000.. Train loss: 0.231.. Validation loss: 0.231
Epoch 636/1000.. Train loss: 0.228.. Validation loss: 0.232
Epoch 637/1000.. Train loss: 0.241.. Validation loss: 0.232
Epoch 638/1000.. Train loss: 0.240.. Validation loss: 0.231
Epoch 639/1000.. Train loss: 0.231.. Validation loss: 0.234
Epoch 640/1000.. Train loss: 0.236.. Validation loss: 0.231
Epoch 641/1000.. Train loss: 0.228.. Validation loss: 0.234
Epoch 642/1000.. Train loss: 0.233.. Validation loss: 0.235
Epoch 643/1000.. Train loss: 0.221.. Validation loss: 0.237
Epoch 644/1000.. Train loss: 0.232.. Validation loss: 0.236
Epoch 645/1000.. Train loss: 0.225.. Validation loss: 0.234
Epoch 646/1000.. Train loss: 0.239.. Validation loss: 0.234
Epoch 647/1000.. Train loss: 0.232.. Validation loss: 0.234
Epoch 648/1000.. Train loss: 0.234.. Validation loss: 0.234
Epoch 649/1000.. Train loss: 0.239.. Validation loss: 0.235
Epoch 650/1000.. Train loss: 0.229.. Validation loss: 0.233
Epoch 651/1000.. Train loss: 0.228.. Validation loss: 0.231
Epoch 652/1000.. Train loss: 0.223.. Validation loss: 0.233
Epoch 653/1000.. Train loss: 0.227.. Validation loss: 0.235
Epoch 654/1000.. Train loss: 0.231.. Validation loss: 0.232
Epoch 655/1000.. Train loss: 0.244.. Validation loss: 0.231
Epoch 656/1000.. Train loss: 0.233.. Validation loss: 0.234
Epoch 657/1000.. Train loss: 0.225.. Validation loss: 0.233
Epoch 658/1000.. Train loss: 0.237.. Validation loss: 0.233
Epoch 659/1000.. Train loss: 0.234.. Validation loss: 0.240
Epoch 660/1000.. Train loss: 0.236.. Validation loss: 0.233
Epoch 661/1000.. Train loss: 0.232.. Validation loss: 0.233
Epoch 662/1000.. Train loss: 0.222.. Validation loss: 0.237
Epoch 663/1000.. Train loss: 0.226.. Validation loss: 0.234
Epoch 664/1000.. Train loss: 0.227.. Validation loss: 0.232
Epoch 665/1000.. Train loss: 0.228.. Validation loss: 0.231
Epoch 666/1000.. Train loss: 0.236.. Validation loss: 0.232
Epoch 667/1000.. Train loss: 0.222.. Validation loss: 0.233
Epoch 668/1000.. Train loss: 0.234.. Validation loss: 0.236
Epoch 669/1000.. Train loss: 0.245.. Validation loss: 0.230
Epoch 670/1000.. Train loss: 0.233.. Validation loss: 0.233
Epoch 671/1000.. Train loss: 0.241.. Validation loss: 0.233
Epoch 672/1000.. Train loss: 0.231.. Validation loss: 0.233
Epoch 673/1000.. Train loss: 0.228.. Validation loss: 0.231
Epoch 674/1000.. Train loss: 0.238.. Validation loss: 0.233
Epoch 675/1000.. Train loss: 0.226.. Validation loss: 0.233
Epoch 676/1000.. Train loss: 0.240.. Validation loss: 0.233
Epoch 677/1000.. Train loss: 0.227.. Validation loss: 0.231
Epoch 678/1000.. Train loss: 0.215.. Validation loss: 0.230
Epoch 679/1000.. Train loss: 0.239.. Validation loss: 0.235
Epoch 680/1000.. Train loss: 0.235.. Validation loss: 0.232
Epoch 681/1000.. Train loss: 0.244.. Validation loss: 0.232
Epoch 682/1000.. Train loss: 0.229.. Validation loss: 0.232
Epoch 683/1000.. Train loss: 0.230.. Validation loss: 0.238
Epoch 684/1000.. Train loss: 0.224.. Validation loss: 0.236
Epoch 685/1000.. Train loss: 0.242.. Validation loss: 0.233
Epoch 686/1000.. Train loss: 0.238.. Validation loss: 0.238
Epoch 687/1000.. Train loss: 0.227.. Validation loss: 0.235
Epoch 688/1000.. Train loss: 0.234.. Validation loss: 0.231
Epoch 689/1000.. Train loss: 0.242.. Validation loss: 0.232
Epoch 690/1000.. Train loss: 0.235.. Validation loss: 0.230
Epoch 691/1000.. Train loss: 0.230.. Validation loss: 0.236
Epoch 692/1000.. Train loss: 0.221.. Validation loss: 0.229
Epoch 693/1000.. Train loss: 0.233.. Validation loss: 0.235
Epoch 694/1000.. Train loss: 0.238.. Validation loss: 0.233
Epoch 695/1000.. Train loss: 0.237.. Validation loss: 0.233
Epoch 696/1000.. Train loss: 0.235.. Validation loss: 0.230
Epoch 697/1000.. Train loss: 0.228.. Validation loss: 0.230
Epoch 698/1000.. Train loss: 0.222.. Validation loss: 0.232
Epoch 699/1000.. Train loss: 0.233.. Validation loss: 0.236
Epoch 700/1000.. Train loss: 0.239.. Validation loss: 0.229
Epoch 701/1000.. Train loss: 0.237.. Validation loss: 0.232
Epoch 702/1000.. Train loss: 0.221.. Validation loss: 0.231
Epoch 703/1000.. Train loss: 0.225.. Validation loss: 0.233
Epoch 704/1000.. Train loss: 0.242.. Validation loss: 0.229
Epoch 705/1000.. Train loss: 0.231.. Validation loss: 0.233
Epoch 706/1000.. Train loss: 0.236.. Validation loss: 0.233
Epoch 707/1000.. Train loss: 0.235.. Validation loss: 0.231
Epoch 708/1000.. Train loss: 0.226.. Validation loss: 0.234
Epoch 709/1000.. Train loss: 0.228.. Validation loss: 0.232
Epoch 710/1000.. Train loss: 0.230.. Validation loss: 0.232
Epoch 711/1000.. Train loss: 0.216.. Validation loss: 0.230
Epoch 712/1000.. Train loss: 0.229.. Validation loss: 0.233
Epoch 713/1000.. Train loss: 0.230.. Validation loss: 0.231
Epoch 714/1000.. Train loss: 0.227.. Validation loss: 0.231
Epoch 715/1000.. Train loss: 0.227.. Validation loss: 0.234
Epoch 716/1000.. Train loss: 0.229.. Validation loss: 0.239
Epoch 717/1000.. Train loss: 0.220.. Validation loss: 0.232
Epoch 718/1000.. Train loss: 0.234.. Validation loss: 0.232
Epoch 719/1000.. Train loss: 0.227.. Validation loss: 0.235
Epoch 720/1000.. Train loss: 0.225.. Validation loss: 0.238
Epoch 721/1000.. Train loss: 0.238.. Validation loss: 0.231
Epoch 722/1000.. Train loss: 0.227.. Validation loss: 0.236
Epoch 723/1000.. Train loss: 0.227.. Validation loss: 0.233
Epoch 724/1000.. Train loss: 0.225.. Validation loss: 0.234
Epoch 725/1000.. Train loss: 0.234.. Validation loss: 0.230
Epoch 726/1000.. Train loss: 0.228.. Validation loss: 0.233
Epoch 727/1000.. Train loss: 0.222.. Validation loss: 0.232
Epoch 728/1000.. Train loss: 0.227.. Validation loss: 0.241
Epoch 729/1000.. Train loss: 0.224.. Validation loss: 0.236
Epoch 730/1000.. Train loss: 0.235.. Validation loss: 0.231
Epoch 731/1000.. Train loss: 0.231.. Validation loss: 0.234
Epoch 732/1000.. Train loss: 0.233.. Validation loss: 0.236
Epoch 733/1000.. Train loss: 0.232.. Validation loss: 0.230
Epoch 734/1000.. Train loss: 0.223.. Validation loss: 0.233
Epoch 735/1000.. Train loss: 0.227.. Validation loss: 0.238
Epoch 736/1000.. Train loss: 0.236.. Validation loss: 0.234
Epoch 737/1000.. Train loss: 0.239.. Validation loss: 0.228
Epoch 738/1000.. Train loss: 0.240.. Validation loss: 0.230
Epoch 739/1000.. Train loss: 0.225.. Validation loss: 0.232
Epoch 740/1000.. Train loss: 0.217.. Validation loss: 0.232
Epoch 741/1000.. Train loss: 0.229.. Validation loss: 0.229
Epoch 742/1000.. Train loss: 0.227.. Validation loss: 0.236
Epoch 743/1000.. Train loss: 0.228.. Validation loss: 0.233
Epoch 744/1000.. Train loss: 0.229.. Validation loss: 0.233
Epoch 745/1000.. Train loss: 0.231.. Validation loss: 0.237
Epoch 746/1000.. Train loss: 0.234.. Validation loss: 0.230
Epoch 747/1000.. Train loss: 0.236.. Validation loss: 0.232
Epoch 748/1000.. Train loss: 0.228.. Validation loss: 0.232
Epoch 749/1000.. Train loss: 0.232.. Validation loss: 0.236
Epoch 750/1000.. Train loss: 0.235.. Validation loss: 0.230
Epoch 751/1000.. Train loss: 0.227.. Validation loss: 0.239
Epoch 752/1000.. Train loss: 0.234.. Validation loss: 0.236
Epoch 753/1000.. Train loss: 0.221.. Validation loss: 0.233
Epoch 754/1000.. Train loss: 0.239.. Validation loss: 0.234
Epoch 755/1000.. Train loss: 0.233.. Validation loss: 0.233
Epoch 756/1000.. Train loss: 0.224.. Validation loss: 0.231
Epoch 757/1000.. Train loss: 0.234.. Validation loss: 0.236
Epoch 758/1000.. Train loss: 0.217.. Validation loss: 0.242
Epoch 759/1000.. Train loss: 0.223.. Validation loss: 0.248
Epoch 760/1000.. Train loss: 0.233.. Validation loss: 0.230
Epoch 761/1000.. Train loss: 0.223.. Validation loss: 0.239
Epoch 762/1000.. Train loss: 0.239.. Validation loss: 0.238
Epoch 763/1000.. Train loss: 0.227.. Validation loss: 0.233
Epoch 764/1000.. Train loss: 0.231.. Validation loss: 0.230
Epoch 765/1000.. Train loss: 0.226.. Validation loss: 0.230
Epoch 766/1000.. Train loss: 0.228.. Validation loss: 0.231
Epoch 767/1000.. Train loss: 0.223.. Validation loss: 0.231
Epoch 768/1000.. Train loss: 0.226.. Validation loss: 0.234
Epoch 769/1000.. Train loss: 0.237.. Validation loss: 0.231
Epoch 770/1000.. Train loss: 0.225.. Validation loss: 0.236
Epoch 771/1000.. Train loss: 0.220.. Validation loss: 0.231
Epoch 772/1000.. Train loss: 0.225.. Validation loss: 0.231
Epoch 773/1000.. Train loss: 0.229.. Validation loss: 0.233
Epoch 774/1000.. Train loss: 0.226.. Validation loss: 0.235
Epoch 775/1000.. Train loss: 0.226.. Validation loss: 0.231
Epoch 776/1000.. Train loss: 0.235.. Validation loss: 0.230
Epoch 777/1000.. Train loss: 0.216.. Validation loss: 0.240
Epoch 778/1000.. Train loss: 0.227.. Validation loss: 0.231
Epoch 779/1000.. Train loss: 0.219.. Validation loss: 0.234
Epoch 780/1000.. Train loss: 0.239.. Validation loss: 0.232
Epoch 781/1000.. Train loss: 0.215.. Validation loss: 0.238
Epoch 782/1000.. Train loss: 0.226.. Validation loss: 0.242
Epoch 783/1000.. Train loss: 0.227.. Validation loss: 0.234
Epoch 784/1000.. Train loss: 0.234.. Validation loss: 0.233
Epoch 785/1000.. Train loss: 0.226.. Validation loss: 0.232
Epoch 786/1000.. Train loss: 0.239.. Validation loss: 0.227
Epoch 787/1000.. Train loss: 0.223.. Validation loss: 0.232
Epoch 788/1000.. Train loss: 0.230.. Validation loss: 0.232
Epoch 789/1000.. Train loss: 0.242.. Validation loss: 0.229
Epoch 790/1000.. Train loss: 0.230.. Validation loss: 0.238
Epoch 791/1000.. Train loss: 0.227.. Validation loss: 0.227
Epoch 792/1000.. Train loss: 0.223.. Validation loss: 0.233
Epoch 793/1000.. Train loss: 0.239.. Validation loss: 0.229
Epoch 794/1000.. Train loss: 0.227.. Validation loss: 0.231
Epoch 795/1000.. Train loss: 0.219.. Validation loss: 0.235
Epoch 796/1000.. Train loss: 0.222.. Validation loss: 0.229
Epoch 797/1000.. Train loss: 0.219.. Validation loss: 0.232
Epoch 798/1000.. Train loss: 0.235.. Validation loss: 0.232
Epoch 799/1000.. Train loss: 0.236.. Validation loss: 0.231
Epoch 800/1000.. Train loss: 0.230.. Validation loss: 0.238
Epoch 801/1000.. Train loss: 0.226.. Validation loss: 0.234
Epoch 802/1000.. Train loss: 0.231.. Validation loss: 0.229
Epoch 803/1000.. Train loss: 0.243.. Validation loss: 0.228
Epoch 804/1000.. Train loss: 0.232.. Validation loss: 0.230
Epoch 805/1000.. Train loss: 0.219.. Validation loss: 0.235
Epoch 806/1000.. Train loss: 0.227.. Validation loss: 0.234
Epoch 807/1000.. Train loss: 0.227.. Validation loss: 0.230
Epoch 808/1000.. Train loss: 0.232.. Validation loss: 0.232
Epoch 809/1000.. Train loss: 0.235.. Validation loss: 0.231
Epoch 810/1000.. Train loss: 0.218.. Validation loss: 0.229
Epoch 811/1000.. Train loss: 0.223.. Validation loss: 0.234
Epoch 812/1000.. Train loss: 0.226.. Validation loss: 0.229
Epoch 813/1000.. Train loss: 0.229.. Validation loss: 0.230
Epoch 814/1000.. Train loss: 0.220.. Validation loss: 0.232
Epoch 815/1000.. Train loss: 0.220.. Validation loss: 0.227
Epoch 816/1000.. Train loss: 0.229.. Validation loss: 0.230
Epoch 817/1000.. Train loss: 0.237.. Validation loss: 0.231
Epoch 818/1000.. Train loss: 0.236.. Validation loss: 0.232
Epoch 819/1000.. Train loss: 0.223.. Validation loss: 0.232
Epoch 820/1000.. Train loss: 0.227.. Validation loss: 0.231
Epoch 821/1000.. Train loss: 0.220.. Validation loss: 0.228
Epoch 822/1000.. Train loss: 0.221.. Validation loss: 0.234
Epoch 823/1000.. Train loss: 0.235.. Validation loss: 0.228
Epoch 824/1000.. Train loss: 0.233.. Validation loss: 0.231
Epoch 825/1000.. Train loss: 0.228.. Validation loss: 0.229
Epoch 826/1000.. Train loss: 0.227.. Validation loss: 0.230
Epoch 827/1000.. Train loss: 0.227.. Validation loss: 0.232
Epoch 828/1000.. Train loss: 0.228.. Validation loss: 0.226
Epoch 829/1000.. Train loss: 0.224.. Validation loss: 0.230
Epoch 830/1000.. Train loss: 0.228.. Validation loss: 0.227
Epoch 831/1000.. Train loss: 0.222.. Validation loss: 0.228
Epoch 832/1000.. Train loss: 0.230.. Validation loss: 0.232
Epoch 833/1000.. Train loss: 0.218.. Validation loss: 0.230
Epoch 834/1000.. Train loss: 0.232.. Validation loss: 0.225
Epoch 835/1000.. Train loss: 0.217.. Validation loss: 0.225
Epoch 836/1000.. Train loss: 0.232.. Validation loss: 0.228
Epoch 837/1000.. Train loss: 0.234.. Validation loss: 0.226
Epoch 838/1000.. Train loss: 0.233.. Validation loss: 0.228
Epoch 839/1000.. Train loss: 0.233.. Validation loss: 0.230
Epoch 840/1000.. Train loss: 0.229.. Validation loss: 0.227
Epoch 841/1000.. Train loss: 0.223.. Validation loss: 0.232
Epoch 842/1000.. Train loss: 0.227.. Validation loss: 0.224
Epoch 843/1000.. Train loss: 0.228.. Validation loss: 0.227
Epoch 844/1000.. Train loss: 0.233.. Validation loss: 0.227
Epoch 845/1000.. Train loss: 0.227.. Validation loss: 0.227
Epoch 846/1000.. Train loss: 0.227.. Validation loss: 0.224
Epoch 847/1000.. Train loss: 0.235.. Validation loss: 0.227
Epoch 848/1000.. Train loss: 0.227.. Validation loss: 0.223
Epoch 849/1000.. Train loss: 0.229.. Validation loss: 0.226
Epoch 850/1000.. Train loss: 0.226.. Validation loss: 0.225
Epoch 851/1000.. Train loss: 0.237.. Validation loss: 0.225
Epoch 852/1000.. Train loss: 0.226.. Validation loss: 0.227
Epoch 853/1000.. Train loss: 0.226.. Validation loss: 0.228
Epoch 854/1000.. Train loss: 0.219.. Validation loss: 0.225
Epoch 855/1000.. Train loss: 0.233.. Validation loss: 0.228
Epoch 856/1000.. Train loss: 0.234.. Validation loss: 0.223
Epoch 857/1000.. Train loss: 0.228.. Validation loss: 0.228
Epoch 858/1000.. Train loss: 0.228.. Validation loss: 0.228
Epoch 859/1000.. Train loss: 0.232.. Validation loss: 0.227
Epoch 860/1000.. Train loss: 0.240.. Validation loss: 0.223
Epoch 861/1000.. Train loss: 0.226.. Validation loss: 0.223
Epoch 862/1000.. Train loss: 0.227.. Validation loss: 0.230
Epoch 863/1000.. Train loss: 0.225.. Validation loss: 0.226
Epoch 864/1000.. Train loss: 0.227.. Validation loss: 0.232
Epoch 865/1000.. Train loss: 0.223.. Validation loss: 0.227
Epoch 866/1000.. Train loss: 0.224.. Validation loss: 0.228
Epoch 867/1000.. Train loss: 0.222.. Validation loss: 0.225
Epoch 868/1000.. Train loss: 0.227.. Validation loss: 0.223
Epoch 869/1000.. Train loss: 0.217.. Validation loss: 0.225
Epoch 870/1000.. Train loss: 0.216.. Validation loss: 0.233
Epoch 871/1000.. Train loss: 0.222.. Validation loss: 0.222
Epoch 872/1000.. Train loss: 0.224.. Validation loss: 0.227
Epoch 873/1000.. Train loss: 0.226.. Validation loss: 0.227
Epoch 874/1000.. Train loss: 0.233.. Validation loss: 0.236
Epoch 875/1000.. Train loss: 0.216.. Validation loss: 0.226
Epoch 876/1000.. Train loss: 0.219.. Validation loss: 0.231
Epoch 877/1000.. Train loss: 0.225.. Validation loss: 0.228
Epoch 878/1000.. Train loss: 0.235.. Validation loss: 0.227
Epoch 879/1000.. Train loss: 0.220.. Validation loss: 0.230
Epoch 880/1000.. Train loss: 0.232.. Validation loss: 0.238
Epoch 881/1000.. Train loss: 0.225.. Validation loss: 0.234
Epoch 882/1000.. Train loss: 0.223.. Validation loss: 0.231
Epoch 883/1000.. Train loss: 0.236.. Validation loss: 0.227
Epoch 884/1000.. Train loss: 0.221.. Validation loss: 0.238
Epoch 885/1000.. Train loss: 0.218.. Validation loss: 0.224
Epoch 886/1000.. Train loss: 0.226.. Validation loss: 0.227
Epoch 887/1000.. Train loss: 0.229.. Validation loss: 0.235
Epoch 888/1000.. Train loss: 0.221.. Validation loss: 0.226
Epoch 889/1000.. Train loss: 0.227.. Validation loss: 0.228
Epoch 890/1000.. Train loss: 0.217.. Validation loss: 0.228
Epoch 891/1000.. Train loss: 0.233.. Validation loss: 0.226
Epoch 892/1000.. Train loss: 0.237.. Validation loss: 0.230
Epoch 893/1000.. Train loss: 0.221.. Validation loss: 0.229
Epoch 894/1000.. Train loss: 0.223.. Validation loss: 0.238
Epoch 895/1000.. Train loss: 0.224.. Validation loss: 0.234
Epoch 896/1000.. Train loss: 0.220.. Validation loss: 0.230
Epoch 897/1000.. Train loss: 0.219.. Validation loss: 0.225
Epoch 898/1000.. Train loss: 0.230.. Validation loss: 0.229
Epoch 899/1000.. Train loss: 0.227.. Validation loss: 0.229
Epoch 900/1000.. Train loss: 0.218.. Validation loss: 0.227
Epoch 901/1000.. Train loss: 0.226.. Validation loss: 0.234
Epoch 902/1000.. Train loss: 0.234.. Validation loss: 0.229
Epoch 903/1000.. Train loss: 0.214.. Validation loss: 0.235
Epoch 904/1000.. Train loss: 0.221.. Validation loss: 0.235
Epoch 905/1000.. Train loss: 0.220.. Validation loss: 0.232
Epoch 906/1000.. Train loss: 0.222.. Validation loss: 0.228
Epoch 907/1000.. Train loss: 0.228.. Validation loss: 0.233
Epoch 908/1000.. Train loss: 0.227.. Validation loss: 0.229
Epoch 909/1000.. Train loss: 0.235.. Validation loss: 0.227
Epoch 910/1000.. Train loss: 0.222.. Validation loss: 0.229
Epoch 911/1000.. Train loss: 0.233.. Validation loss: 0.227
Epoch 912/1000.. Train loss: 0.222.. Validation loss: 0.230
Epoch 913/1000.. Train loss: 0.223.. Validation loss: 0.229
Epoch 914/1000.. Train loss: 0.216.. Validation loss: 0.231
Epoch 915/1000.. Train loss: 0.225.. Validation loss: 0.231
Epoch 916/1000.. Train loss: 0.235.. Validation loss: 0.222
Epoch 917/1000.. Train loss: 0.220.. Validation loss: 0.230
Epoch 918/1000.. Train loss: 0.225.. Validation loss: 0.236
Epoch 919/1000.. Train loss: 0.221.. Validation loss: 0.230
Epoch 920/1000.. Train loss: 0.221.. Validation loss: 0.231
Epoch 921/1000.. Train loss: 0.227.. Validation loss: 0.229
Epoch 922/1000.. Train loss: 0.225.. Validation loss: 0.231
Epoch 923/1000.. Train loss: 0.224.. Validation loss: 0.228
Epoch 924/1000.. Train loss: 0.211.. Validation loss: 0.236
Epoch 925/1000.. Train loss: 0.224.. Validation loss: 0.233
Epoch 926/1000.. Train loss: 0.215.. Validation loss: 0.241
Epoch 927/1000.. Train loss: 0.224.. Validation loss: 0.231
Epoch 928/1000.. Train loss: 0.233.. Validation loss: 0.229
Epoch 929/1000.. Train loss: 0.237.. Validation loss: 0.237
Epoch 930/1000.. Train loss: 0.218.. Validation loss: 0.238
Epoch 931/1000.. Train loss: 0.218.. Validation loss: 0.233
Epoch 932/1000.. Train loss: 0.239.. Validation loss: 0.234
Epoch 933/1000.. Train loss: 0.226.. Validation loss: 0.229
Epoch 934/1000.. Train loss: 0.224.. Validation loss: 0.225
Epoch 935/1000.. Train loss: 0.229.. Validation loss: 0.225
Epoch 936/1000.. Train loss: 0.220.. Validation loss: 0.242
Epoch 937/1000.. Train loss: 0.219.. Validation loss: 0.234
Epoch 938/1000.. Train loss: 0.238.. Validation loss: 0.229
Epoch 939/1000.. Train loss: 0.219.. Validation loss: 0.227
Epoch 940/1000.. Train loss: 0.226.. Validation loss: 0.236
Epoch 941/1000.. Train loss: 0.221.. Validation loss: 0.232
Epoch 942/1000.. Train loss: 0.221.. Validation loss: 0.229
Epoch 943/1000.. Train loss: 0.213.. Validation loss: 0.231
Epoch 944/1000.. Train loss: 0.232.. Validation loss: 0.225
Epoch 945/1000.. Train loss: 0.224.. Validation loss: 0.237
Epoch 946/1000.. Train loss: 0.220.. Validation loss: 0.224
Epoch 947/1000.. Train loss: 0.216.. Validation loss: 0.228
Epoch 948/1000.. Train loss: 0.217.. Validation loss: 0.221
Epoch 949/1000.. Train loss: 0.223.. Validation loss: 0.224
Epoch 950/1000.. Train loss: 0.216.. Validation loss: 0.227
Epoch 951/1000.. Train loss: 0.221.. Validation loss: 0.229
Epoch 952/1000.. Train loss: 0.214.. Validation loss: 0.232
Epoch 953/1000.. Train loss: 0.217.. Validation loss: 0.225
Epoch 954/1000.. Train loss: 0.233.. Validation loss: 0.230
Epoch 955/1000.. Train loss: 0.225.. Validation loss: 0.225
Epoch 956/1000.. Train loss: 0.212.. Validation loss: 0.228
Epoch 957/1000.. Train loss: 0.224.. Validation loss: 0.232
Epoch 958/1000.. Train loss: 0.227.. Validation loss: 0.231
Epoch 959/1000.. Train loss: 0.220.. Validation loss: 0.225
Epoch 960/1000.. Train loss: 0.218.. Validation loss: 0.225
Epoch 961/1000.. Train loss: 0.221.. Validation loss: 0.227
Epoch 962/1000.. Train loss: 0.222.. Validation loss: 0.232
Epoch 963/1000.. Train loss: 0.232.. Validation loss: 0.234
Epoch 964/1000.. Train loss: 0.223.. Validation loss: 0.227
Epoch 965/1000.. Train loss: 0.219.. Validation loss: 0.226
Epoch 966/1000.. Train loss: 0.227.. Validation loss: 0.233
Epoch 967/1000.. Train loss: 0.230.. Validation loss: 0.227
Epoch 968/1000.. Train loss: 0.213.. Validation loss: 0.230
Epoch 969/1000.. Train loss: 0.227.. Validation loss: 0.231
Epoch 970/1000.. Train loss: 0.229.. Validation loss: 0.223
Epoch 971/1000.. Train loss: 0.224.. Validation loss: 0.236
Epoch 972/1000.. Train loss: 0.220.. Validation loss: 0.234
Epoch 973/1000.. Train loss: 0.213.. Validation loss: 0.223
Epoch 974/1000.. Train loss: 0.226.. Validation loss: 0.235
Epoch 975/1000.. Train loss: 0.222.. Validation loss: 0.232
Epoch 976/1000.. Train loss: 0.221.. Validation loss: 0.233
Epoch 977/1000.. Train loss: 0.224.. Validation loss: 0.231
Epoch 978/1000.. Train loss: 0.237.. Validation loss: 0.232
Epoch 979/1000.. Train loss: 0.225.. Validation loss: 0.238
Epoch 980/1000.. Train loss: 0.226.. Validation loss: 0.222
Epoch 981/1000.. Train loss: 0.216.. Validation loss: 0.227
Epoch 982/1000.. Train loss: 0.223.. Validation loss: 0.232
Epoch 983/1000.. Train loss: 0.229.. Validation loss: 0.231
Epoch 984/1000.. Train loss: 0.225.. Validation loss: 0.230
Epoch 985/1000.. Train loss: 0.225.. Validation loss: 0.235
Epoch 986/1000.. Train loss: 0.223.. Validation loss: 0.223
Epoch 987/1000.. Train loss: 0.221.. Validation loss: 0.229
Epoch 988/1000.. Train loss: 0.227.. Validation loss: 0.231
Epoch 989/1000.. Train loss: 0.212.. Validation loss: 0.247
Epoch 990/1000.. Train loss: 0.227.. Validation loss: 0.231
Epoch 991/1000.. Train loss: 0.219.. Validation loss: 0.228
Epoch 992/1000.. Train loss: 0.212.. Validation loss: 0.237
Epoch 993/1000.. Train loss: 0.225.. Validation loss: 0.230
Epoch 994/1000.. Train loss: 0.225.. Validation loss: 0.240
Epoch 995/1000.. Train loss: 0.223.. Validation loss: 0.238
Epoch 996/1000.. Train loss: 0.219.. Validation loss: 0.237
Epoch 997/1000.. Train loss: 0.208.. Validation loss: 0.232
Epoch 998/1000.. Train loss: 0.214.. Validation loss: 0.246
Epoch 999/1000.. Train loss: 0.226.. Validation loss: 0.230
Epoch 1000/1000.. Train loss: 0.223.. Validation loss: 0.237"""

lines = input_string.split("\n")

global_training_losses = []
global_validation_losses = []

for line in lines:
    parts = line.split("..")
    train_loss = float(parts[1].split(":")[1])
    validation_loss = float(parts[2].split(":")[1])
    
    global_training_losses.append(train_loss)
    global_validation_losses.append(validation_loss)

print("Training losses:", global_training_losses)
print("Validation losses:", global_validation_losses)




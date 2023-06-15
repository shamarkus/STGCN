import torch
import torch.nn.functional as F
from torch import nn

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
    def __init__(self, in_channels, out_channels, A):
        super(STGCN, self).__init__()

        self.pixel_dimension = in_channels
        self.joint_dimension = A.shape[0]
        self.output_dimension = out_channels

        self.stgcn_blocks = nn.Sequential(
            STGCNBlock(in_channels, 32, A, residual = False),
            STGCNBlock(32, 32, A),
            STGCNBlock(32, 32, A),
            STGCNBlock(32, 64, A, stride=(2, 1)),  # Set stride to 2 for 4th layer
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A, stride=(2, 1)),  # Set stride to 2 for 7th layer
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
    

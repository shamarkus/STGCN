import torch
import torch.nn.functional as F
from torch import nn
from .GraphConvolution import *

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=(1, 1), residual=False):
        super(STGCNBlock, self).__init__()

        self.spatial_gc = GraphConvolution(in_channels, out_channels)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == (1, 1)):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride),
                nn.BatchNorm2d(out_channels),
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

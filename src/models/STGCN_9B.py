import torch
from torch import nn
from .STGCNBlock import *

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
    

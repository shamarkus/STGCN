import torch
import sys

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for joint_positions, lengths, label in dataloader:
        # Move tensors to the configured device
        joint_positions = joint_positions.to(device)
        label = label.to(device).view(-1,1)

        # Forward pass
        outputs = model(joint_positions, lengths)

        if torch.isnan(outputs[0][0]):
            sys.exit()

        # l1_lambda = 0.00
        # l1_norm = sum(p.abs().sum() for p in model.parameters())
        loss = criterion(outputs, label) # + l1_lambda * l1_norm

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * joint_positions.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

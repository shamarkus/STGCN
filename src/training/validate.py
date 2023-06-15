import torch

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for joint_positions, lengths, label in dataloader:
            # Move tensors to the configured device
            joint_positions = joint_positions.to(device).reshape(-1, joint_positions.size(1), int(joint_positions.size(2) / 3), 3)
            label = label.to(device).view(-1, 1)

            # Forward pass
            outputs = model(joint_positions, lengths)
           
            l1_lambda = 0.00
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = criterion(outputs, label) + l1_lambda * l1_norm

            running_loss += loss.item() * joint_positions.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


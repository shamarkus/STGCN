import torch
import sys

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for joint_positions, lengths, label in dataloader:
        # Move tensors to the configured device
        joint_positions = joint_positions.to(device).reshape(-1, joint_positions.size(1), int(joint_positions.size(2) / 3), 3)
        label = label.to(device).view(-1,1)

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

import torch
import os
from torch import nn
from .train import train
from .validate import validate
from .utils import getDataloaders

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


# ES_dataset should be a string, m01, m02, m10 or es1...es5
def trainModel(model, es_str, device, lr, bs, num_epochs, train_loader = None, val_loader = None, fold = None):
	global_training_losses = []
	global_validation_losses = []
  
	# Define the loss function and optimizer
	criterion = nn.MSELoss()
	# criterion = nn.L1Loss()

	optimizer = torch.optim.Adam(model.parameters(), lr = lr)

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.3)
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.2, patience = 4, threshold = 0.001)
	# scheduler = CustomLRAdjuster(optimizer, threshold=0.2, factor=0.1)

	# Get the dataloaders
	if train_loader == None or val_loader == None:
		train_loader, val_loader = getDataloaders(es_str, bs)

	# Get path for saving models
	models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))

	# Training Loop
	for epoch in range(num_epochs):
		train_loss = train(model, train_loader, criterion, optimizer, device)

		val_loss = validate(model, val_loader, criterion, device)

		scheduler.step()
		# scheduler.step(val_loss)

		global_training_losses.append(train_loss)
		global_validation_losses.append(val_loss)

	# Save model if validation loss is less than 0.150
		if val_loss < 0.140:
			if fold == None:
				torch.save(model.state_dict(), os.path.join(models_path,f"{es_str}_{epoch+1}_{train_loss:.4f}_{val_loss:.4f}_model.pth"))
			else:
				torch.save(model.state_dict(), os.path.join(models_path,f"{es_str}_FOLD{fold}_E{epoch+1}_T{train_loss:.4f}_V{val_loss:.4f}_model.pth"))


		print(f"Epoch {epoch+1}/{num_epochs}.. Train loss: {train_loss:.3f}.. Validation loss: {val_loss:.3f}", flush=True)


	return global_training_losses, global_validation_losses 

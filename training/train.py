import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:  #assign tensors to device
        inputs, targets = input.to(device) 

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Loss: {loss.item()}')

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f'Epoch {i+1}')
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print('-----------------------')
    print('Training is done')

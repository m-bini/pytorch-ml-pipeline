# file to create your custom loss function

from torch import nn

def custom_loss_function():
    return nn.CrossEntropyLoss()
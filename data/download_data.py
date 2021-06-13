from torchvision import datasets
from torchvision.transforms import ToTensor

# download dataset and create dataloader

def download_mnist_datasets():
    train_data = datasets.MNIST(
        root='data',
        download=True,
        train=True,
        transform=ToTensor()
    )
    validation_data = datasets.MNIST(
        root='data',
        download=True,
        train=False,
        transform=ToTensor()
    )
    return train_data, validation_data
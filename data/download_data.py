from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# download dataset and create dataloader
def download_mnist_datasets():
    train_data = MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    validation_data = MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
    return train_data, validation_data
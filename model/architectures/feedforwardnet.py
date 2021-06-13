from torch import nn

# create model

class FeedForwardNet(nn.Module):

    def __init__(self, parameters):
        super().__init__()

        # in_dim = 784, hidden_dim = 256, out_dim = 10
        self.in_dim = parameters['in_dim']
        self.hidden_dim = parameters['hidden_dim']
        self.out_dim = parameters['out_dim']
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.flatten(input_data)
        x = self.dense_layers(x)
        x = self.softmax(x)
        return x
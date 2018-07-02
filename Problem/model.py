import torch.nn as nn


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size):

        super(MultiLayerPerceptron, self).__init__()

        self.input_size = input_size
        self.output_size = 1

        self.fc1 = nn.Linear(self.input_size, 64)
        self.nonlin1 = nn.Tanh()
        self.fc2 = nn.Linear(64, 1)
        self.out = nn.Sigmoid()

    def forward(self, features):

        x = self.nonlin1(self.fc1(features))
        x = self.fc2(x)

        return self.out(x)
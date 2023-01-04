import torch
import torch.nn as nn

class MappingV2A(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        # self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(input_dim, output_dim*2)
        # self.bn2 = nn.BatchNorm1d(output_dim*2)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(output_dim*2, output_dim)

        # self.scaler = nn.Linear(1, 1, bias=False)

    def forward(self, x):
        x = self.fc(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x



import torch
import torch.nn as nn


class BaseModel(nn.Module):
    # Two layers MLP
    def __init__(self, input_dim, num_classes, hidden_dim = 256, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.fc2(self.activation(self.fc1(x)))
    
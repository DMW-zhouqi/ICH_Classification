# Import related package
import  torch
from    torch import nn

# Defines a class that flattens the data
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape) # Returns the flattened data

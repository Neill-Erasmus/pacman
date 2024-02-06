from torch.nn import functional as F
from torch import nn
import torch

class NeuralNetwork(nn.Module):
    """
    NeuralNetwork class defines a convolutional neural network architecture for an RL agent.

    Args:
        action_size (int): Dimension of the action space.
        seed (int, optional): Random seed. Default is 42.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization layer for conv1.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization layer for conv2.
        conv3 (nn.Conv2d): Third convolutional layer.
        bn3 (nn.BatchNorm2d): Batch normalization layer for conv3.
        conv4 (nn.Conv2d): Fourth convolutional layer.
        bn4 (nn.BatchNorm2d): Batch normalization layer for conv4.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
    """

    def __init__(self, action_size : int, seed : int = 42) -> None:
        """
        Initializes a neural network instance with given action size and seed.

        Args:
            action_size (int): Dimension of the action space.
            seed (int, optional): Random seed. Default is 42.
        """

        super(NeuralNetwork, self).__init__()
        self.seed  : torch.Generator = torch.manual_seed(seed=seed)
        self.conv1 : nn.Conv2d       = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.bn1   : nn.BatchNorm2d  = nn.BatchNorm2d(num_features=self.conv1.out_channels)
        self.conv2 : nn.Conv2d       = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=64, kernel_size=4, stride=2)
        self.bn2   : nn.BatchNorm2d  = nn.BatchNorm2d(num_features=self.conv2.out_channels)
        self.conv3 : nn.Conv2d       = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=64, kernel_size=3, stride=1)
        self.bn3   : nn.BatchNorm2d  = nn.BatchNorm2d(num_features=self.conv3.out_channels)
        self.conv4 : nn.Conv2d       = nn.Conv2d(in_channels=self.conv3.out_channels, out_channels=128, kernel_size=3, stride=1)
        self.bn4   : nn.BatchNorm2d  = nn.BatchNorm2d(num_features=self.conv4.out_channels)
        self.fc1   : nn.Linear       = nn.Linear(in_features=(10*10*128), out_features=512)
        self.fc2   : nn.Linear       = nn.Linear(in_features=self.fc1.out_features, out_features=256)
        self.fc3   : nn.Linear       = nn.Linear(in_features=self.fc2.out_features, out_features=action_size)

    def forward(self, state : torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            state (torch.Tensor): Input state.

        Returns:
            torch.Tensor: Output Q-values for each action.
        """

        x : torch.Tensor = F.relu(self.bn1(self.conv1(state)))
        x : torch.Tensor = F.relu(self.bn2(self.conv2(x)))
        x : torch.Tensor = F.relu(self.bn3(self.conv3(x)))
        x : torch.Tensor = F.relu(self.bn4(self.conv4(x)))
        x : torch.Tensor = x.view(x.size(0), -1)
        x : torch.Tensor = F.relu(self.fc1(x))
        x : torch.Tensor = F.relu(self.fc2(x))
        return self.fc3(x)
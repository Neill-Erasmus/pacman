from neural_network import NeuralNetwork
from hyperparameters import Hyperparameters
from torch.nn import functional as F
from torchvision import transforms
from typing import Tuple, List
from collections import deque
from torch import optim
from PIL import Image
import numpy as np
import random
import torch

parameters: Hyperparameters = Hyperparameters()

def preprocess_frame(frame : np.ndarray) -> torch.Tensor:
    """
    Preprocess a frame by resizing and converting to a PyTorch tensor.

    Args:
        frame (np.ndarray): Input frame.

    Returns:
        torch.Tensor: Preprocessed frame.
    """

    frame                           = Image.fromarray(frame) #type: ignore
    preprocess : transforms.Compose = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    return preprocess(frame).unsqueeze(0)

class Agent:
    """
    Deep Q-Learning Agent.

    Args:
        action_size (int): Number of possible actions.

    Attributes:
        device (torch.device): Device (CPU or GPU) for training the agent.
        action_size (int): Number of possible actions.
        local_qnet (NeuralNetwork): Local Q-network for the agent.
        target_qnet (NeuralNetwork): Target Q-network for stability during training.
        optimizer (torch.optim): Optimizer for updating the Q-network.
        memory (deque): Replay memory for storing and sampling experiences.
    """

    def __init__(self, action_size : int) -> None:
        """
        Initializes the Agent object with the given action size.

        Args:
            action_size (int): Number of possible actions.
        """

        self.device      : torch.device    = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.action_size : int             = action_size
        self.local_qnet  : NeuralNetwork   = NeuralNetwork(action_size).to(self.device)
        self.target_qnet : NeuralNetwork   = NeuralNetwork(action_size).to(self.device)
        self.optimizer   : optim.Optimizer = optim.Adam(self.local_qnet.parameters(), lr=parameters.learning_rate)
        self.memory      : deque           = deque(maxlen=10000)

    def step(self, state : np.ndarray, action : int, reward : float, next_state : np.ndarray, done : bool) -> None:
        """
        Take a step in the environment and store the experience.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Flag indicating episode termination.
        """

        state      = preprocess_frame(state) #type: ignore
        next_state = preprocess_frame(next_state) #type: ignore
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > parameters.minibatch_size:
            experiences: List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]] = random.sample(self.memory, k=parameters.minibatch_size)
            self.learn(experiences, parameters.gamma)

    def action(self, state, epsilon: float = 0.) -> int:
        """
        Select an action using an epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state.
            epsilon (float): Exploration-exploitation trade-off parameter.

        Returns:
            int: Selected action.
        """

        state = preprocess_frame(state).to(self.device)
        self.local_qnet.eval()
        with torch.no_grad():
            action_values: torch.Tensor = self.local_qnet(state)
        self.local_qnet.train()
        if random.random() > epsilon:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return int(random.choice(np.arange(self.action_size)))

    def learn(self, experiences: List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]], gamma: float) -> None:
        """
        Update the Q-network based on experiences sampled from the replay memory.

        Args:
            experiences (List[Tuple[torch.Tensor, int, float, torch.Tensor, bool]]): List of experiences.
            gamma (float): Discount factor for future rewards.
        """

        states, actions, rewards, next_states, dones = zip(*experiences)
        states         = torch.cat(states).float().to(self.device)
        actions        = torch.tensor(actions).long().to(self.device)
        rewards        = torch.tensor(rewards).float().to(self.device)
        next_states    = torch.cat(next_states).float().to(self.device)
        dones          = torch.tensor(dones).float().to(self.device)
        q_expected     = self.local_qnet(states).gather(1, actions.view(-1, 1))
        next_q_targets = self.target_qnet(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets      = rewards.unsqueeze(1) + (gamma * next_q_targets * (1 - dones.unsqueeze(1)))
        loss           = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
# Pacman

A Deep Convolutional Q-Network (DCQN) has been developed and trained to master the Pacman environment within OpenAI's Gymnasium. This project is dedicated to harnessing reinforcement learning methodologies, empowering an agent to independently navigate and successfully play the game of Pacman.

## Deep Convolutional Q-Learning

### Q-Learning

Q-learning is a model-free reinforcement learning algorithm used to learn the quality of actions in a given state. It operates by learning a Q-function, which represents the expected cumulative reward of taking a particular action in a particular state and then following a policy to maximize cumulative rewards thereafter.

### Deep Learning

Deep learning involves using neural networks with multiple layers to learn representations of data. Convolutional neural networks (CNNs) are a specific type of neural network commonly used for analyzing visual imagery. They are composed of layers such as convolutional layers, pooling layers, and fully connected layers, which are designed to extract features from images.

### Combining Q-Learning with Deep Learning

In traditional Q-learning, a Q-table is used to store the Q-values for all state-action pairs. However, for complex environments with large state spaces, maintaining such a table becomes infeasible due to memory constraints. Deep Q-Networks address this issue by using a neural network to approximate the Q-function, mapping states to Q-values directly from raw input pixels.

### Deep Convolutional Q-Learning

Deep Convolutional Q-Learning specifically utilizes CNNs as function approximators within the Q-learning framework. It's particularly effective in environments where the state space is represented as visual input, such as playing Atari games from raw pixel inputs.

The general steps involved in training a Deep Convolutional Q-Learning agent are as follows:

#### Observation

The agent observes the environment, typically represented as images or raw sensory data.
Action Selection: Based on the observed state, the agent selects an action according to an exploration strategy, such as ε-greedy.

#### Reward

The agent receives a reward from the environment based on the action taken.

#### Experience Replay

The agent stores its experiences (state, action, reward, next state) in a replay buffer.

#### Training

Periodically, the agent samples experiences from the replay buffer and uses them to update the parameters of the neural network using techniques like stochastic gradient descent (SGD) or variants like RMSprop or Adam.

#### Target Network

To stabilize training, a separate target network may be used to calculate target Q-values during updates. This network is periodically updated with the parameters from the primary network.

#### Iteration

The process of observation, action selection, and training continues iteratively until convergence.
By leveraging deep convolutional neural networks, Deep Convolutional Q-Learning has demonstrated remarkable success in learning effective control policies directly from high-dimensional sensory input, making it a powerful technique for solving complex reinforcement learning problems, especially in the realm of visual tasks like playing video games or robotic control.

## Overview of Pacman Environment

<p align="center">
  <img src="https://github.com/Neill-Erasmus/pacman/assets/141222943/3eb890b6-5143-41e1-9630-064e966790cf" alt="video">
</p>

### Description

The Pacman environment is a classic arcade game where the player controls Pacman, a character moving around a maze, eating food pellets while avoiding ghosts. When Pacman consumes a Power Pellet, it gains the ability to eat the ghosts.

### Action Space

The action space for Pacman is Discrete(5), meaning there are five possible actions Pacman can take. These actions include moving up, down, left, right, or taking no action (NOOP). Optionally, with full_action_space=True, all 18 possible actions that can be performed on an Atari 2600 can be enabled.

### Observation Space

The observations in the Pacman environment can be of three types: "rgb", "grayscale", and "ram".
If "rgb" is chosen, the observation space is Box(0, 255, (210, 160, 3)), representing a color image with three color channels (RGB).
If "ram" is chosen, the observation space is Box(0, 255, (128,)), representing the RAM (Random Access Memory) of the Atari 2600.
If "grayscale" is chosen, the observation space is Box(0, 255, (210, 160)), representing a grayscale version of the RGB observation.

## The Architecture of the Neural Network

### Convolutional Layers

The network consists of four convolutional layers (conv1, conv2, conv3, and conv4). These layers are responsible for learning hierarchical features from the input image (state). Each convolutional layer applies a set of filters to the input image to extract relevant features. The output channels for each convolutional layer are 32, 64, 64, and 128, respectively. The kernel sizes for the convolutional layers are 8x8, 4x4, 3x3, and 3x3, and the strides are 4, 2, 1, and 1, respectively.

### Batch Normalization Layers

Batch normalization layers (bn1, bn2, bn3, and bn4) are applied after each convolutional layer to normalize the activations. This helps in stabilizing and accelerating the training process by reducing internal covariate shift.

### Fully Connected Layers

Following the convolutional layers, there are three fully connected layers (fc1, fc2, and fc3). These layers are responsible for transforming the high-dimensional feature maps from the convolutional layers into Q-values for each action. The first fully connected layer has 512 output features, the second has 256 output features, and the third has action_size output features, which corresponds to the dimension of the action space.

### Activation Function

ReLU (Rectified Linear Unit) activation function is applied after each layer, except for the output layer. ReLU introduces non-linearity to the network, allowing it to learn complex relationships in the data.

### Forward Pass

In the forward method, the input state is passed through the convolutional layers with ReLU activation and batch normalization applied at each step. Then, the output is flattened (view) to be fed into the fully connected layers. Finally, the Q-values for each action are obtained as the output of the last fully connected layer (fc3).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

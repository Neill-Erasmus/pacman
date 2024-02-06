class Hyperparameters():
    """
    Hyperparameters class stores the hyperparameters for training an RL agent.

    Attributes:
        learning_rate (float): The learning rate for the optimizer. Default is 5e-4.
        minibatch_size (int): Size of the minibatch for training. Default is 64.
        gamma (float): Discount factor for future rewards. Default is 0.99.
    """

    def __init__(self) -> None:
        """
        Initializes the Hyperparameters object with default values.
        """

        self.learning_rate  : float = 5e-4
        self.minibatch_size : int   = 64
        self.gamma          : float = 0.99
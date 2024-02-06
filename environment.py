import gym

class Environment:
    """
    Environment class wraps the gym environment for the RL agent.

    Attributes:
        env (gym.Env): The gym environment instance.
        state_shape (tuple): The shape of the state space.
        state_size (int): The size of the state space.
        number_actions (int): The number of actions in the action space.
    """

    def __init__(self) -> None:
        """
        Initializes the Environment object with a specific gym environment ('MsPacmanDeterministic-v0').
        """

        self.env            : gym.Env = gym.make('MsPacmanDeterministic-v0', full_action_space=False)
        self.state_shape    : tuple   = self.env.observation_space.shape #type: ignore
        self.state_size     : int     = self.state_shape[0]              #type: ignore
        self.number_actions : int     = self.env.action_space.n          #type: ignore
        print(f'State Shape: {self.state_shape}\nState Size: {self.state_size}\nNumber of Actions: {self.number_actions}')

    def step(self, action : int) -> tuple:
        """
        Performs a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            Tuple containing state, reward, done, and additional info.
        """

        return self.env.step(action)

    def reset(self) -> object:
        """
        Resets the environment.

        Returns:
            Initial state after reset.
        """

        return self.env.reset()
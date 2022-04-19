import numpy as np


class RandomPolicy():
    def __init__(self,
                 state_dims,
                 num_actions):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        # TODO: implement here
        self.s_dim = state_dims
        self.a_dim = num_actions

    def __call__(self,s) -> int:
        # TODO: implement this method
        action = np.random.choice(self.a_dim)
        return action

    def update(self, s, a, gamma_t, delta):
        pass


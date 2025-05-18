import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        """
        Args:
            state_dim (tuple): State shape, e.g., (4, 84, 84)
            action_dim (int or tuple): Action shape (usually scalar for discrete actions)
            max_size (int): Maximum buffer size
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.s = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.a = np.zeros((max_size, *action_dim) if isinstance(action_dim, tuple) else (max_size, 1), dtype=np.int64)
        self.r = np.zeros((max_size, 1), dtype=np.float32)
        self.s_prime = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.terminated = np.zeros((max_size, 1), dtype=np.float32)

    def update(self, s, a, r, s_prime, terminated):
        """
        Store a transition in the buffer.
        """
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_prime[self.ptr] = s_prime
        self.terminated[self.ptr] = terminated

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        Returns:
            tuple of tensors: (states, actions, rewards, next_states, dones)
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        return (self.s[ind],self.a[ind],self.r[ind],self.s_prime[ind],self.terminated[ind],)

    def __len__(self):
        return self.size

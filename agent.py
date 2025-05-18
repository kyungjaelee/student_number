import numpy as np
import torch
import torch.nn.functional as F
from neural_network import MLPQNetwork
from buffer import ReplayBuffer
from exploration import EpsilonGreedy, SoftmaxExploration

class Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        exploration_strategy=SoftmaxExploration(),
        lr=0.00025,
        gamma=0.99,
        batch_size=64,
        warmup_steps=5000,
        buffer_size=int(1e5),
        target_update_interval=5000
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.target_update_interval = target_update_interval

        self.network = MLPQNetwork(state_dim, action_dim)
        self.target_network = MLPQNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr)
        self.buffer = ReplayBuffer(state_dim, (1,), buffer_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        self.target_network.to(self.device)

        self.total_steps = 0
        self.exploration_strategy = exploration_strategy
        self.exploration_strategy.action_dim = action_dim  # if needed

    @torch.no_grad()
    def act(self, state):
        if self.total_steps < self.warmup_steps:
            return np.random.randint(0, self.action_dim)

        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q = self.network(x)
        q_np = q.cpu().numpy().squeeze()
        return self.exploration_strategy.select_action(q_np)
            
    @torch.no_grad()
    def select_best_action(self, state):
        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        q = self.network(x)
        return int(torch.argmax(q, dim=1).item())
    
    def learn(self):
        s, a, r, s_prime, terminated = self.buffer.sample(self.batch_size)
        
        # Convert numpy arrays to torch tensors
        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        r = torch.from_numpy(r).float().to(self.device)
        s_prime = torch.from_numpy(s_prime).float().to(self.device)
        terminated = torch.from_numpy(terminated).float().to(self.device)
        
        with torch.no_grad():
            next_q = self.target_network(s_prime)
            max_next_q = next_q.max(dim=1, keepdim=True).values
            td_target = r + (1. - terminated) * self.gamma * max_next_q

        q_values = self.network(s).gather(1, a.long())
        loss = F.mse_loss(q_values, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"total_steps": self.total_steps, "value_loss": loss.item()}

    def process(self, transition):
        result = {}
        self.total_steps += 1
        self.buffer.update(*transition)

        if self.total_steps > self.warmup_steps:
            result = self.learn()

        if self.total_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        self.exploration_strategy.update(self.total_steps)
        return result

    def train_epoch(self, env, epoch_steps):
        states, _ = env.reset(seed=42)
        num_envs = env.num_envs
        rets = np.zeros(num_envs)

        step_count = 0
        while step_count < epoch_steps:  # safety cap
            actions = [self.act(states[i]) for i in range(num_envs)]
            next_states, rewards, terminateds, truncateds, infos = env.step(actions)
            dones = np.logical_or(terminateds, truncateds)

            for i in range(num_envs):
                transition = (
                    states[i],
                    actions[i],
                    rewards[i],
                    next_states[i],
                    float(dones[i])
                )
                self.process(transition)
                rets[i] += rewards[i]

            if np.any(dones):
                states, _ = env.reset(seed=42)
                rets = np.zeros(num_envs)
            else:
                states = next_states

            step_count += num_envs  # because num_envs transitions per step

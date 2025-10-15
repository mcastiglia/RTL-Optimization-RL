from preprocess import generate_prefix_adder_graph, compute_area, compute_power, generate_rca_graph, generate_kogge_stone_graph, generate_sklansky_graph, generate_brent_kung_graph, get_real_area_power, compute_delay_levels
import random
import gym
import numpy as np
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models import Graphormer
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, dense_to_sparse
from stable_baselines3.common.policies import ActorCriticPolicy

class GraphPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.Tanh, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, *args, **kwargs)
        n = 16
        self.n = n
        self.gnn = Graphormer(num_node_types=None, num_edge_types=1, num_classes=None, embed_dim=64)
        self.policy_net = nn.Linear(64, action_space.n)
        self.value_net = nn.Linear(64, 1)

    def forward(self, obs):
        # obs shape: (batch, n*4 + n*n)
        batch_size = obs.shape[0]
        embeddings = []
        for i in range(batch_size):
            x = obs[i, :self.n * 4].view(self.n, 4)
            adj_flat = obs[i, self.n * 4:].view(self.n, self.n)
            edge_index = dense_to_sparse(adj_flat)[0]
            data = Data(x=x, edge_index=edge_index)
            graph_emb = self.gnn(data)
            embeddings.append(graph_emb)
        graph_emb = torch.stack(embeddings)
        action_logits = self.policy_net(graph_emb)
        value = self.value_net(graph_emb)
        return action_logits, value

class AdderEnv(gym.Env):
    def __init__(self):
        super(AdderEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(5)  # 5 adder types
        self.observation_space = gym.spaces.Box(low=4, high=64, shape=(1,), dtype=np.float32)
        self.generators = [generate_rca_graph, generate_prefix_adder_graph, generate_kogge_stone_graph, generate_sklansky_graph, generate_brent_kung_graph]
        self.bit_width = None

    def reset(self):
        self.bit_width = random.randint(4, 64)
        return np.array([self.bit_width], dtype=np.float32)

    def step(self, action):
        g = self.generators[action](self.bit_width)
        area = compute_area(g)
        #power = compute_power(g)
        reward = - area# Negative weighted sum for minimization
        done = True  # Single-step episode
        return np.array([self.bit_width], dtype=np.float32), reward, done, {}

class CircuitGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, labels):
        self.data_list = []
        for graph, label in zip(graphs, labels):
            graph.y = torch.tensor(label)
            self.data_list.append(graph)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

import concurrent.futures

# Supervised training of Graphormer to predict area and power
generators = [generate_rca_graph, generate_prefix_adder_graph, generate_kogge_stone_graph, generate_sklansky_graph, generate_brent_kung_graph]
train_graphs = []
for _ in range(100):
    gen = random.choice(generators)
    n = 16  # Fixed for training
    g = gen(n)
    train_graphs.append(g)

# Generate accurate labels using synthesis (fast flow for intermediate)
def get_label(g):
    return get_real_area_power(g, 'fast')

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    train_labels = list(executor.map(get_label, train_graphs))

train_dataset = CircuitGraphDataset(train_graphs, train_labels)

embed_dim = 768
model = Graphormer(num_node_types=None, num_edge_types=1, num_classes=None, embed_dim=embed_dim)
predictor = torch.nn.Linear(embed_dim, 1)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=2e-5, weight_decay=0.01)

# Training loop
for epoch in range(10):
    model.train()
    predictor.train()
    for batch in train_loader:
        optimizer.zero_grad()
        embedding = model(batch)
        out = predictor(embedding)
        loss = F.mse_loss(out, batch.y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed")

# Now use the trained predictor in RL
class AdderEnv(gym.Env):
    def __init__(self, predictor_model, graph_model, n_bits=16, max_steps=10):
        super(AdderEnv, self).__init__()
        self.n = n_bits
        self.max_steps = max_steps
        self.action_space = gym.spaces.Discrete(self.n * self.n * 2)  # add/remove edge between i,j
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n * 4 + self.n * self.n,), dtype=np.float32)
        self.predictor = predictor_model
        self.graph_model = graph_model
        self.graph_model.eval()
        self.predictor.eval()
        self.reset()

    def reset(self):
        self.current_graph = generate_rca_graph(self.n)  # Start with RCA
        self.update_features()  # Ensure features are up to date
        self.step_count = 0
        self.current_cost = self.compute_cost('full')  # Use full flow for initial
        return self.get_obs()

    def get_obs(self):
        # Get adjacency matrix
        adj = torch.zeros(self.n, self.n)
        adj[self.current_graph.edge_index[0], self.current_graph.edge_index[1]] = 1
        # Flatten x and adj
        obs = torch.cat([self.current_graph.x.flatten(), adj.flatten()])
        return obs.numpy().astype(np.float32)

    def update_features(self):
        edge_index = self.current_graph.edge_index
        n = self.n
        fan_in = degree(edge_index[1], num_nodes=n).float()
        fan_out = degree(edge_index[0], num_nodes=n).float()
        delay_levels = compute_delay_levels(edge_index, n)
        bit_positions = torch.arange(n, dtype=torch.float)
        self.current_graph.x = torch.stack([bit_positions, fan_in, fan_out, delay_levels], dim=1)

    def compute_cost(self, flow_type='fast'):
        real_area = get_real_area_power(self.current_graph, flow_type)
        return real_area 

    def step(self, action):
        # Decode action: add_remove = action // (n*n), i = (action % (n*n)) // n, j = action % n
        total = self.n * self.n
        add_remove = action // total
        idx = action % total
        i = idx // self.n
        j = idx % self.n
        flow_type = 'full' if self.step_count == 0 or self.step_count == self.max_steps - 1 else 'fast'
        if i == j:
            # No self-loops
            reward = 0
        else:
            edge_exists = (self.current_graph.edge_index[0] == i).any() and (self.current_graph.edge_index[1] == j).any()
            if add_remove == 0:  # remove
                if edge_exists:
                    # Remove edge
                    mask = ~((self.current_graph.edge_index[0] == i) & (self.current_graph.edge_index[1] == j))
                    self.current_graph.edge_index = self.current_graph.edge_index[:, mask]
                    self.update_features()
                    new_cost = self.compute_cost(flow_type)
                    reward = self.current_cost - new_cost
                    self.current_cost = new_cost
                else:
                    reward = 0
            else:  # add
                if not edge_exists:
                    # Add edge
                    new_edge = torch.tensor([[i], [j]], dtype=torch.long)
                    self.current_graph.edge_index = torch.cat([self.current_graph.edge_index, new_edge], dim=1)
                    self.update_features()
                    new_cost = self.compute_cost(flow_type)
                    reward = self.current_cost - new_cost
                    self.current_cost = new_cost
                else:
                    reward = 0
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return self.get_obs(), reward, done, {}

# Load RL model
env = AdderEnv(predictor, model)
rl_model = PPO(GraphPolicy, env, verbose=1)

# Training
rl_model.learn(total_timesteps=10000)

# Evaluation
total_reward = 0
num_episodes = 20
for _ in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = rl_model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    total_reward += episode_reward
print(f"Average reward: {total_reward / num_episodes}")
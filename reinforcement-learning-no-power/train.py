from preprocess import generate_prefix_adder_graph, compute_area, compute_power, generate_rca_graph, generate_kogge_stone_graph, generate_sklansky_graph, generate_brent_kung_graph, get_real_area_power, compute_delay_levels
import random
import gym
import numpy as np
from stable_baselines3 import PPO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree, dense_to_sparse
from stable_baselines3.common.policies import ActorCriticPolicy
import os
import json
import time
import matplotlib

class GCNEncoder(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=256, out_dim=768, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, data):
        # data.x: [num_nodes_total, in_channels]
        # data.edge_index: [2, num_edges_total]
        # data.batch: [num_nodes_total]  (present when batching)
        x, edge_index = data.x, data.edge_index
        x = self.act(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.act(self.conv2(x, edge_index))
        # pool to graph-level: returns [batch_size, out_dim]
        batch = getattr(data, 'batch', None)
        if batch is None:
            # single graph: average nodes
            return x.mean(dim=0, keepdim=True)  # [1, out_dim]
        return global_mean_pool(x, batch)       # [batch_size, out_dim]
    

class GraphPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.Tanh, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, *args, **kwargs)
        n = 16
        self.n = n
        self.gnn = GCNEncoder(in_channels=4, hidden_dim=128, out_dim=64, dropout=0.1)
        self.policy_net = nn.Linear(64, action_space.n)
        self.value_net = nn.Linear(64, 1)

    def forward(self, obs):
        # obs shape: (batch, n*4 + n*n)
        # Stable-Baselines3 expects forward to return (actions, values, log_probs)
        # Ensure obs is a torch tensor on the right device
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        obs = obs.to(self.device)

        batch_size = obs.shape[0]
        embeddings = []
        for i in range(batch_size):
            x = obs[i, :self.n * 4].view(self.n, 4)
            adj_flat = obs[i, self.n * 4:].view(self.n, self.n)
            edge_index = dense_to_sparse(adj_flat)[0]
            data = Data(x=x, edge_index=edge_index)
            data = data.to(self.device)
            graph_emb = self.gnn(data)            # graph_emb shape: [1, out_dim]
            graph_emb = graph_emb.squeeze(0)     # -> [out_dim]
            embeddings.append(graph_emb)

        graph_emb = torch.stack(embeddings)      # [batch, out_dim]
        action_logits = self.policy_net(graph_emb)  # [batch, action_dim]
        value = self.value_net(graph_emb).squeeze(-1)   # [batch]

        # Build action distribution and sample / pick deterministically
        dist = torch.distributions.Categorical(logits=action_logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, value, log_probs

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
model = GCNEncoder(in_channels=4, hidden_dim=256, out_dim=embed_dim, dropout=0.1)
predictor = torch.nn.Linear(embed_dim, 1)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
predictor = predictor.to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=2e-5)

# Training loop
# Ensure output directory exists for saving logs and graphs
os.makedirs('output_data', exist_ok=True)
log_path = os.path.join('output_data', 'training_log.txt')
with open(log_path, 'a') as f:
    f.write(f"=== Supervised training started: {time.asctime()} ===\n")

for epoch in range(10):
    model.train()
    predictor.train()
    epoch_loss = 0.0
    batch_count = 0
    for batch in train_loader:
        batch = batch.to(device)           # if using device
        batch.y = batch.y.float()          # ensure float
        # optionally ensure batch.y shape: [batch_size, 1]
        if batch.y.dim() == 1:
            batch.y = batch.y.unsqueeze(1)
        optimizer.zero_grad()
        embedding = model(batch)
        out = predictor(embedding)
        loss = F.mse_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batch_count += 1
    avg_loss = epoch_loss / batch_count if batch_count else 0.0
    with open(log_path, 'a') as f:
        f.write(f"Epoch {epoch+1} avg_loss={avg_loss:.6f}\n")
    print(f"Epoch {epoch+1} completed (avg_loss={avg_loss:.6f})")

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
        # Use predictor to estimate area (predictor expects graph_model embeddings)
        self.graph_model.eval()
        self.predictor.eval()
        data = self.current_graph
        # move data (x, edge_index) to device and create a Batch if needed
        data = data.to(next(self.graph_model.parameters()).device)
        with torch.no_grad():
            emb = self.graph_model(data)           # returns [1, embed_dim] or [batch, embed_dim]
            pred_area = self.predictor(emb).item()
        return pred_area 

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
for episode_idx in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = rl_model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
    total_reward += episode_reward
    # after each episode finishes, save the final graph
    # Create per-episode directory
    ep_dir = os.path.join('output_data', f'ep{episode_idx}')
    os.makedirs(ep_dir, exist_ok=True)
    graph_pt = os.path.join(ep_dir, f"final_graph_ep{episode_idx}.pt")
    graph_json = os.path.join(ep_dir, f"final_graph_ep{episode_idx}.json")
    torch.save(env.current_graph, graph_pt)
    data = {
        "edge_index": env.current_graph.edge_index.tolist(),
        "x": env.current_graph.x.tolist()
    }
    with open(graph_json, 'w') as f:
        json.dump(data, f)
    with open(log_path, 'a') as f:
        f.write(f"Episode {episode_idx} reward={episode_reward:.6f} saved:{graph_pt},{graph_json}\n")

avg_reward = total_reward / num_episodes
with open(log_path, 'a') as f:
    f.write(f"=== Evaluation completed: {time.asctime()} avg_reward={avg_reward:.6f} ===\n")
print(f"Average reward: {avg_reward}")

# Save trained models
model_path = os.path.join('output_data', 'gnn_encoder.pt')
predictor_path = os.path.join('output_data', 'predictor.pt')
torch.save(model.state_dict(), model_path)
torch.save(predictor.state_dict(), predictor_path)
with open(log_path, 'a') as f:
    f.write(f"Saved model weights: {model_path}, {predictor_path}\n")

# Save RL agent (Stable-Baselines3) if available
try:
    rl_model_path = os.path.join('output_data', 'rl_model.zip')
    rl_model.save(rl_model_path)
    with open(log_path, 'a') as f:
        f.write(f"Saved RL model: {rl_model_path}\n")
except Exception as e:
    with open(log_path, 'a') as f:
        f.write(f"Could not save RL model: {e}\n")

# Export final graphs to PNG using networkx + matplotlib
try:
    import networkx as nx
    import matplotlib.pyplot as plt
    for episode_idx in range(num_episodes):
        json_path = os.path.join('output_data', f'ep{episode_idx}', f"final_graph_ep{episode_idx}.json")
        try:
            with open(json_path, 'r') as jf:
                gjson = json.load(jf)
        except Exception:
            # JSON not available, skip
            continue
        edge_idx = gjson.get('edge_index', [])
        num_nodes = len(gjson.get('x', []))
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        if edge_idx and len(edge_idx) == 2:
            G.add_edges_from(list(zip(edge_idx[0], edge_idx[1])))
        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=300)
        png_path = os.path.join('output_data', f'ep{episode_idx}', f"final_graph_ep{episode_idx}.png")
        plt.savefig(png_path)
        plt.close()
        with open(log_path, 'a') as f:
            f.write(f"Saved graph image: {png_path}\n")
except Exception as e:
    with open(log_path, 'a') as f:
        f.write(f"Failed to export graph images: {e}\n")
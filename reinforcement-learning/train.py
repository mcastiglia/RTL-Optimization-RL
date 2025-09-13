import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.models import Graphormer
from torch_geometric.loader import DataLoader
from preprocess import generate_prefix_adder_graph, compute_area, compute_power, generate_rca_graph, generate_kogge_stone_graph, generate_sklansky_graph, generate_brent_kung_graph, get_real_area_power
import random

# Assume you have a dataset class that returns graphormer_input and labels
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

# Prepare your dataset
# Generate training data with mix of adder types
generators = [generate_rca_graph, generate_prefix_adder_graph, generate_kogge_stone_graph, generate_sklansky_graph, generate_brent_kung_graph]
train_graphs = []
for _ in range(100):
    gen = random.choice(generators)
    n = random.randint(4, 8)
    g = gen(n)
    train_graphs.append(g)
train_labels = [[compute_area(g), compute_power(g)] for g in train_graphs]

eval_graphs = []
for _ in range(20):
    gen = random.choice(generators)
    n = random.randint(4, 8)
    g = gen(n)
    eval_graphs.append(g)
eval_labels = [[compute_area(g), compute_power(g)] for g in eval_graphs]

train_dataset = CircuitGraphDataset(train_graphs, train_labels)
eval_dataset = CircuitGraphDataset(eval_graphs, eval_labels)

# Load model for regression
embed_dim = 768
model = Graphormer(num_node_types=None, num_edge_types=1, num_classes=None, embed_dim=embed_dim)
predictor = torch.nn.Linear(embed_dim, 2)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.01)

# Training loop
step = 0
for epoch in range(10):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        embedding = model(batch)
        out = predictor(embedding)
        loss = F.mse_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        
        step += 1
        if step % 50 == 0:
            # Every 50 steps, get real synthesis estimates for a random training sample
            idx = random.randint(0, len(train_graphs) - 1)
            real_area, real_power = get_real_area_power(train_graphs[idx])
            est_area, est_power = train_labels[idx]
            print(f"Step {step}: Real vs Estimated - Area: {real_area:.2f} vs {est_area:.2f}, Power: {real_power:.2f} vs {est_power:.2f}")
        
    print(f"Epoch {epoch+1} completed")

# Evaluation
model.eval()
total_mse = 0
count = 0
with torch.no_grad():
    for batch in eval_loader:
        embedding = model(batch)
        out = predictor(embedding)
        mse = F.mse_loss(out, batch.y)
        total_mse += mse.item()
        count += 1
print(f"Average MSE: {total_mse / count}")
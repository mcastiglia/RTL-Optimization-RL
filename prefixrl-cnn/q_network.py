
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union, Dict
import numpy as np
import torch
import torch.nn as nn
    
# Residual block for the CNN (Adapted from PrefixRL Figure 2)
class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, negative_slope: float = 0.01, norm: str = "bn", gn_groups: int = 32):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.act1 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.act_out = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        out = self.act_out(out)
        return out

# Q-Network architecture (Adapted from PrefixRL Figure 2)
# Authors use a block size of 32, channel width of 256
class PrefixRL_DQN(nn.Module):
    def __init__(
        self,
        width: int = 256,
        blocks: int = 32,
        stem_channels: Optional[int] = None,
        negative_slope: float = 0.01,
        norm: str = "bn",
        gn_groups: int = 32,
        in_channels: int = 4,
        out_channels: int = 4,
    ):
        super().__init__()
        c = stem_channels or width

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

        body = []
        for _ in range(blocks):
            body.append(ResidualBlock(c, kernel_size=5, negative_slope=negative_slope, norm=norm, gn_groups=gn_groups))
        self.body = nn.Sequential(*body)

        self.head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(c, out_channels, kernel_size=1, bias=True),
        )

        self._init_weights(negative_slope)

    def _init_weights(self, negative_slope: float):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=negative_slope, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        y = self.body(y)
        y = self.head(y)
        return y

# Build input tensor features from the nodelist, minlist, levellist, and fanoutlist (Adapted from PrefixRL Section C)
def build_features(
    nodelist: np.ndarray, minlist: np.ndarray, levellist: np.ndarray, fanoutlist: np.ndarray, 
    batch_size: int = 1, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:

    # Normalize the levellist and fanoutlist to [0,1]
    normalized_levellist = normalize_features(levellist, levellist.max())
    normalized_fanoutlist = normalize_features(fanoutlist, fanoutlist.max())

    arr = np.stack([nodelist, minlist, normalized_levellist, normalized_fanoutlist], axis=0)  # (4, N, N)
    arr = np.broadcast_to(arr, (batch_size,) + arr.shape).copy()  # (B, 4, N, N)
    
    return torch.from_numpy(arr).to(device=device, dtype=dtype)

# Normalize the features to [0,1] (Adapted from PrefixRL Section C)
def normalize_features(arr, denom: Optional[float], eps: float = 1e-8, clip: bool = True):
    arr = np.asarray(arr, dtype=np.float32)
    if denom is None:
        m = float(arr.max()) if arr.size > 0 else 1.0
        denom = max(m, eps)
    out = arr.astype(np.float32) / float(denom + eps)
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out

# Initialize an empty mask to set positions and actions that are illegal to true
# Upper triangle, diagonal, left column, and nodelist are illegal for adding nodes
# Upper triangle, diagonal, left column, and minlist are illegal for deleting nodes
def build_action_masks(N: int, nodelist: np.ndarray, minlist: np.ndarray) -> Dict[str, torch.Tensor]:
    all_mask = np.zeros((N, N), dtype=bool)
    nodelist_mask = np.zeros((N, N), dtype=bool)
    minlist_mask = np.zeros((N, N), dtype=bool)
    
    # Upper triangle mask
    iu, ju = np.triu_indices(N, k=1)
    all_mask[iu, ju] = True
    
    # Diagonal mask
    idx = np.arange(N)
    all_mask[idx, idx] = True
    
    # Left column mask
    all_mask[idx, 0] = True
    
    # nodelist mask
    for i in range(N):
        for j in range(N):
            if nodelist[i, j] == 1.0:
                nodelist_mask[i, j] = True
                
    # minlist mask
    for i in range(N):
        for j in range(N):
            if minlist[i, j] == 0.0:
                minlist_mask[i, j] = True
    
    # Create dictionary of masks. all_mask will be ORed with other masks in apply_action_masks
    masks = {
        "all": torch.from_numpy(all_mask),
        "area_add": torch.from_numpy(nodelist_mask),
        "area_del": torch.from_numpy(minlist_mask),
        "delay_add": torch.from_numpy(nodelist_mask),
        "delay_del": torch.from_numpy(minlist_mask),
    }
    return masks

# Apply action masks to the Q-maps (Adapted from PrefixRL Section C)
def apply_action_masks(qmaps: torch.Tensor, masks: Dict[str, torch.Tensor], fill_value: float = -1e9) -> torch.Tensor:
    assert qmaps.dim() == 4 and qmaps.size(1) == 4, "Expected (B,4,N,N)"
    B, _, N, M = qmaps.shape
    assert N == M, "Expected square maps (N==N)."

    # Expand all masks to (B, N, N)
    all_mask = masks["all"].to(device=qmaps.device).bool().expand(B, N, N)
    per_action = [
        masks["area_add"].to(qmaps.device).bool(),
        masks["area_del"].to(qmaps.device).bool(),
        masks["delay_add"].to(qmaps.device).bool(),
        masks["delay_del"].to(qmaps.device).bool(),
    ]

    out = qmaps.clone()
    
    # Set all illegal actions marked as "True" in the all_mask to fill_value
    for a in range(4):
        out[:, a][all_mask] = fill_value
    
    # Set all illegal actions marked as "True" in the per_action mask to fill_value
    for a in range(4):
        if per_action[a].any():
            mask_a = per_action[a].expand(B, N, N)
            out[:, a][mask_a] = fill_value
            
    return out

# Scalarize the Q-maps to get the scores for adding and deleting nodes (Adapted from PrefixRL Section B)
# TODO: PrefixRL says w_area + w_delay = 1.0 and both are non-negative. Different values in w_area and w_delay explore different tradeoffs in area and delay.
def scalarize_q(qmaps: torch.Tensor, w_area: float = 0.5, w_delay: float = 0.5, c_area: float = 1e-3, c_delay: float = 10.0) -> torch.Tensor:
    assert qmaps.dim() == 4 and qmaps.size(1) == 4, "Expected (B,4,N,N)"
    q_area_add = qmaps[:, 0]
    q_area_del = qmaps[:, 1]
    q_delay_add = qmaps[:, 2]
    q_delay_del = qmaps[:, 3]
    
    # Scale the Q-values by the constants c_area and c_delay to adjust for unit differences, then add the weighted Q-values
    s_add = w_area * (c_area * q_area_add) + w_delay * (c_delay * q_delay_add)
    s_del = w_area * (c_area * q_area_del) + w_delay * (c_delay * q_delay_del)
    
    return torch.stack([s_add, s_del], dim=1)


# Argmax the scores to get the best action for adding and deleting nodes (Adapted from PrefixRL Section B)
def argmax_action(qmaps: torch.Tensor, w_area: float = 1.0, w_delay: float = 1.0):
    # [B,2,N,N] tensor represented the scalarized Q-values for adding and deleting nodes
    scores = scalarize_q(qmaps, w_area, w_delay)  # (B,2,N,N)
    
    # best_vals (shape [B,N,N]) is the maximum scalarized Q-value (either add or delete) per cell
    # best_idx is which action gave that max (0 for add, 1 for delete)
    best_vals, best_idx = scores.max(dim=1)       # (B,N,N), (B,N,N)
    
    # best_is_add: boolean interpretation of best_idx (1: add, 0: delete)
    best_is_add = best_idx == 0
    return best_is_add, best_vals

# Extract the best action and coordinates for each batch element
def get_best_action(qmaps: torch.Tensor, w_area: float = 1.0, w_delay: float = 1.0) -> Tuple[Tuple[int, int], bool]:
    best_is_add, best_vals = argmax_action(qmaps, w_area, w_delay)
    
    batch_size = qmaps.size(0)
    N = qmaps.size(2)
    max_per_batch, flat_idx = best_vals.view(batch_size, -1).max(dim=1)
    
    i = torch.div(flat_idx, N, rounding_mode='floor')
    j = flat_idx % N
    
    batch = torch.arange(batch_size, device=best_vals.device)
    is_add = best_is_add[batch, i, j]
    action_coords = torch.stack([i, j], dim=1)
    
    return action_coords, is_add

# TODO: The .random() function does not exist for how I'm using it. Needs to be fixed
def get_random_action(qmaps: torch.Tensor, w_area: float = 1.0, w_delay: float = 1.0):
    batch_size = qmaps.size(0)
    N = qmaps.size(2)
    random_per_batch, flat_idx = qmaps.view(batch_size, -1).random(dim=1)
    
    i = torch.div(flat_idx, N, rounding_mode='floor')
    j = flat_idx % N
    
    batch = torch.arange(batch_size, device=best_vals.device)
    is_add = best_is_add[batch, i, j]
    action_coords = torch.stack([i, j], dim=1)
    
    return action_coords, is_add

# TODO: This was vibe coded. I need to test this and make sure it works
# A replay buffer stores past experiences (state, action, reward, next_state, done), which are subsequently
# randomly sampled to break correlation between consecutive updates, which increases the stability of the network
class ReplayBuffer:
    def __init__(self, capacity: int = 400_000):      # 4e5
        self.capacity = capacity
        self.buf = []
        self.pos = 0

    def __len__(self): return len(self.buf)

    # Add a tuple of the experience to the buffer
    def push(self, s, a_type, i, j, r, s_next, done):
        tup = (s, int(a_type), int(i), int(j), float(r), s_next, bool(done))
        if len(self.buf) < self.capacity:
            self.buf.append(tup)
        else:
            self.buf[self.pos] = tup
        self.pos = (self.pos + 1) % self.capacity

    # Sample a batch of experiences from the buffer
    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, i, j, r, s2, d = zip(*batch)
        s  = torch.stack(s)        # (B,4,N,N)
        s2 = torch.stack(s2)
        a  = torch.tensor(a, dtype=torch.long, device=s.device)
        i  = torch.tensor(i, dtype=torch.long, device=s.device)
        j  = torch.tensor(j, dtype=torch.long, device=s.device)
        r  = torch.tensor(r, dtype=torch.float32, device=s.device)
        d  = torch.tensor(d, dtype=torch.float32, device=s.device)
        return s, a, i, j, r, s2, d
    
# Training parameters from the PrefixRL paper
@dataclass
class TrainingConfig:
    gamma: float = 0.75                      # paper
    lr: float = 4e-5                         # paper
    target_sync_every: int = 60              # paper
    eps_start: float = 1.0
    eps_end: float   = 0.0                   # paper says anneal to 0
    eps_decay_steps: int = 200_000           # choose a schedule; paper doesn’t specify curve
    batch_size: int = 128
    steps: int = 200_000                     # toy; paper runs are much larger
    w_area: float = 0.5                      # you can sweep per-agent
    w_delay: float = 0.5
    c_area: float = 1e-3                      # paper’s scaling
    c_delay: float = 10.0
    
# TODO: This was also vibe coded. There is scaffold code/placeholders where environment actions need to be taken
def train(cfg: TrainingConfig, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TODO: PrefixRL uses a double DQN architecture, where there is an online network (net) and an offline network (tgt)
    # Network actual being trained each step (also called online/policy network)
    net = PrefixRL_DQN(blocks=32, width=256).to(device)  # paper
    
    # Target network: slowly updated copy of the online network, providing stable targets for training
    tgt = PrefixRL_DQN(blocks=32, width=256).to(device)
    tgt.load_state_dict(net.state_dict()); tgt.eval()

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    buf = ReplayBuffer(400_000) # Initialize replay buffer with 4e5 elements

    eps = cfg.eps_start
    eps_slope = (cfg.eps_end - cfg.eps_start) / max(cfg.eps_decay_steps, 1)
    grad_steps = 0

    current_state = init_graph(global_vars.input_bitwidth, global_vars.initial_adder_type)
    
    feats = build_features(init_state.nodelist, init_state.minlist, init_state.levellist, init_state.fanoutlist).to(device)   # (B,4,N,N) in [0,1]

    # Main training loop
    # TODO: Starting with a batch size of 1, need to modify to handle batch size > 1
    for step in range(1, cfg.steps+1):
        # forward + mask
        q = net(feats)
        q = apply_action_masks(q, env.nodelist, env.minlist, env.all_mask, env.add_mask, env.del_mask)

        action_masks = build_action_masks(args.input_bitwidth, init_state.nodelist, init_state.minlist)
        q_masked = apply_action_masks(q, action_masks)
        
        # TODO: ε-greedy on scalarized add/del (choose random actions with probability ε)
        # if random.random() < eps:
        #     best_action_idx, best_action = get_random_action(q_masked, w_area=cfg.w_area, w_delay=cfg.w_delay)
        # else:
        #     best_action_idx, best_action = get_best_action(q_masked, w_area=cfg.w_area, w_delay=cfg.w_delay)
        best_action_idx, best_action = get_best_action(q_masked, w_area=cfg.w_area, w_delay=cfg.w_delay)

        # TODO: Currently, only supports batch size of 1. Need to modify to support batch size > 1
        if (best_action.size(0) == 1):
            action = best_action[0].item()
            action_idx = best_action_idx[0].tolist()
        else:
            raise ValueError("TODO: edit program for batch size > 1")
            
        next_state = init_state.get_next_state(action, action_idx)

        reward = (current_state.area-next_state.area, current_state.delay-next_state.delay)
        
        # reward scalarization per paper (scale raw metrics, then use w)
        # reward is difference between w-optimal points; here, a simple surrogate:
        reward = -(cfg.carea*raw_area + cfg.cdelay*raw_delay)

        feats2 = env.features().to(device)
        buf.push(feats.detach(), a_type, i, j, reward, feats2.detach(), done)

        # learning
        if len(buf) >= cfg.batch_size:
            S, A, I, J, R, S2, D = buf.sample(cfg.batch_size)  # shapes (B,...)
            S  = S.to(device);  S2 = S2.to(device)
            # current Q(s,a)
            q_curr = scalarize_q(apply_action_masks(net(S),  env.nodelist, env.minlist, env.all_mask, env.add_mask, env.del_mask),
                                 cfg.w_area, cfg.w_delay)     # (B,2,N,N)
            B = S.size(0); batch = torch.arange(B, device=device)
            q_sa = q_curr[batch, A, I, J]                     # (B,)

            # target: r + γ max_{a',i',j'} Q_tgt(s',a')
            with torch.no_grad():
                q_next = scalarize_q(apply_action_masks(tgt(S2), env.nodelist, env.minlist, env.all_mask, env.add_mask, env.del_mask),
                                     cfg.w_area, cfg.w_delay) # (B,2,N,N)
                qn_max = q_next.view(B, -1).max(dim=1).values
                target = R + cfg.gamma * (1.0 - D) * qn_max

            loss = F.smooth_l1_loss(q_sa, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            grad_steps += 1
            if grad_steps % cfg.target_sync_every == 0:       # every 60 steps
                tgt.load_state_dict(net.state_dict())

        feats = feats2
        eps = max(cfg.eps_end, eps + eps_slope)

        if done:
            env.reset()
            feats = env.features().to(device)

    return net, tgt
    
    
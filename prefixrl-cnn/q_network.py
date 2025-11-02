
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union, Dict, List
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import global_vars
from init_states import init_graph
from environment import evaluate_next_state
from tqdm import tqdm
import time
    
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

    def _init_weights(self, negative_slope: float) -> None:
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
# TODO: Added support for batching, function could be cleaned up
def build_features(
    nodelist: np.ndarray, minlist: np.ndarray, levellist: np.ndarray, fanoutlist: np.ndarray, 
    batch_size: Optional[int] = None, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:

    if isinstance(nodelist, torch.Tensor):
        nodelist = nodelist.cpu().numpy()
    if isinstance(minlist, torch.Tensor):
        minlist = minlist.cpu().numpy()
    if isinstance(levellist, torch.Tensor):
        levellist = levellist.cpu().numpy()
    if isinstance(fanoutlist, torch.Tensor):
        fanoutlist = fanoutlist.cpu().numpy()
    
    nodelist = np.asarray(nodelist)
    minlist = np.asarray(minlist)
    levellist = np.asarray(levellist)
    fanoutlist = np.asarray(fanoutlist)
    
    is_batched = nodelist.ndim == 3
    
    # If the input is already batched, build features for each batch element
    if is_batched:
        B = nodelist.shape[0]
        if batch_size is not None and B != batch_size:
            raise ValueError(f"Batch size mismatch: input has {B} elements but batch_size={batch_size}")
        
        normalized_levellist = np.zeros_like(levellist, dtype=np.float32)
        normalized_fanoutlist = np.zeros_like(fanoutlist, dtype=np.float32)
        
        for b in range(B):
            levellist_max = levellist[b].max() if levellist[b].size > 0 else 1.0
            fanoutlist_max = fanoutlist[b].max() if fanoutlist[b].size > 0 else 1.0
            normalized_levellist[b] = normalize_features(levellist[b], levellist_max)
            normalized_fanoutlist[b] = normalize_features(fanoutlist[b], fanoutlist_max)
        
        arr = np.stack([nodelist, minlist, normalized_levellist, normalized_fanoutlist], axis=1)
    else:
        if batch_size is None:
            batch_size = 1
        
        normalized_levellist = normalize_features(levellist, levellist.max())
        normalized_fanoutlist = normalize_features(fanoutlist, fanoutlist.max())

        arr = np.stack([nodelist, minlist, normalized_levellist, normalized_fanoutlist], axis=0)
        arr = np.broadcast_to(arr, (batch_size,) + arr.shape).copy()
    
    return torch.from_numpy(arr).to(device=device, dtype=dtype)

# Normalize the features to [0,1] (Adapted from PrefixRL Section C)
def normalize_features(arr, denom: Optional[float], eps: float = 1e-8, clip: bool = True) -> np.ndarray:
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
def apply_action_masks(qmaps: torch.Tensor, masks: Dict[str, torch.Tensor], fill_value: float = -torch.inf) -> torch.Tensor:
    assert qmaps.dim() == 4 and qmaps.size(1) == 4, "Expected (B,4,N,N)"
    B, _, N, M = qmaps.shape
    assert N == M, "Expected square maps (N==N)."

    all_mask_t = masks["all"].to(device=qmaps.device).bool()
    if all_mask_t.ndim == 2:
        all_mask = all_mask_t.expand(B, N, N)
    else:
        all_mask = all_mask_t
    
    per_action = [
        masks["area_add"].to(qmaps.device).bool(),
        masks["area_del"].to(qmaps.device).bool(),
        masks["delay_add"].to(qmaps.device).bool(),
        masks["delay_del"].to(qmaps.device).bool(),
    ]

    out = qmaps.clone()
    
    for a in range(4):
        out[:, a][all_mask] = fill_value
    
    for a in range(4):
        if per_action[a].any():
            mask_a = per_action[a]
            if mask_a.ndim == 2:
                mask_a = mask_a.expand(B, N, N)
            out[:, a][mask_a] = fill_value
            
    return out

# Build and apply action masks to the Q-maps for a batch of states
def build_and_apply_action_masks_batch(qmaps: torch.Tensor, feats: torch.Tensor, N: int) -> torch.Tensor:
    B = qmaps.size(0)
    mask_list = [build_action_masks(N, feats[b,0].cpu().numpy(), feats[b,1].cpu().numpy()) for b in range(B)]

    action_masks = {
        "all": torch.stack([mask_list[b]["all"] for b in range(B)]),
        "area_add": torch.stack([mask_list[b]["area_add"] for b in range(B)]),
        "area_del": torch.stack([mask_list[b]["area_del"] for b in range(B)]),
        "delay_add": torch.stack([mask_list[b]["delay_add"] for b in range(B)]),
        "delay_del": torch.stack([mask_list[b]["delay_del"] for b in range(B)]),
    }
    
    q_masked = apply_action_masks(qmaps, action_masks)
    return q_masked

# Scalarize the Q-maps to get the scores for adding and deleting nodes (Adapted from PrefixRL Section B)
def scalarize_q(qmaps: torch.Tensor, w_area, w_delay, c_area: float = 1e-3, c_delay: float = 10.0) -> torch.Tensor:
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
def argmax_action(qmaps: torch.Tensor, w_area, w_delay) -> Tuple[torch.Tensor, torch.Tensor]:
    # [B,2,N,N] tensor represented the scalarized Q-values for adding and deleting nodes
    scores = scalarize_q(qmaps, w_area, w_delay)  # (B,2,N,N)
    
    # best_vals (shape [B,N,N]) is the maximum scalarized Q-value (either add or delete) per cell
    # best_idx is which action gave that max (0 for add, 1 for delete)
    best_vals, best_idx = scores.max(dim=1)       # (B,N,N), (B,N,N)
    
    # best_is_add: boolean interpretation of best_idx (1: add, 0: delete)
    best_is_add = best_idx == 0
    return best_is_add, best_vals

# Extract the best action and coordinates for each batch element
def get_best_action(qmaps: torch.Tensor, w_area, w_delay) -> Tuple[Tuple[int, int], bool]:
    best_is_add, best_vals = argmax_action(qmaps, w_area, w_delay)
    
    batch_size = qmaps.size(0)
    N = qmaps.size(2)
    max_per_batch, flat_idx = best_vals.view(batch_size, -1).max(dim=1)
    
    i = torch.div(flat_idx, N, rounding_mode='floor')
    j = flat_idx % N
    
    batch = torch.arange(batch_size, device=best_vals.device)
    is_add = best_is_add[batch, i, j]
    action_coords = torch.stack([i, j], dim=1)
    
    return action_coords, is_add, max_per_batch

# Sample an epsilon value for epsilon-greedy exploration
# Adapted from Multi-objective Reinforcement Learning with Adaptive Pareto Reset for Prefix Adder Design, page 11
# TODO: epsilon should anneal to zero over time
def sample_epsilon(episode: int) -> float:
    total_episodes = global_vars.num_episodes
    progress = int(episode / total_episodes)
    i = int((1023 * 0.4) * (1 - (progress * 0.9)))   # linearly decrease i from 1023*0.4 to 0.04 over training
    eps = 0.4 ** (1 + 7 * i / 1023)
    return eps

# Select a random action from the scalarized Q-values
def get_random_action(qmaps: torch.Tensor, w_area, w_delay, fill_value: float = -torch.inf) -> Tuple[Tuple[int, int], bool, torch.Tensor]:
    
    scores = scalarize_q(qmaps, w_area, w_delay)  # (B,2,N,N)
    
    B, C, N, M = scores.shape
    
    scores_mask = scores > fill_value

    # First randomly choose between add (channel 0) and delete (channel 1) with equal probability
    # to ensure balanced exploration between add and delete actions
    action_type_choice = torch.randint(0, 2, (B,), device=scores.device)  # 0 for add, 1 for delete
    
    # Then randomly select from valid actions in the chosen channel
    action_coords = torch.zeros(B, 2, dtype=torch.long, device=scores.device)
    is_add = torch.zeros(B, dtype=torch.bool, device=scores.device)
    rand_per_batch = torch.zeros(B, device=scores.device)
    
    for b in range(B):
        channel_idx = action_type_choice[b]
        channel_mask = scores_mask[b, channel_idx]  # (N, M)
        
        # If chosen channel has no valid actions, try the other channel
        if not channel_mask.any():
            channel_idx = 1 - channel_idx
            channel_mask = scores_mask[b, channel_idx]
        
        # Select a random valid action from the channel
        if channel_mask.any():
            valid_indices = channel_mask.nonzero(as_tuple=False)
            rand_idx = torch.randint(0, len(valid_indices), (1,), device=scores.device)
            i, j = valid_indices[rand_idx][0]
            action_coords[b] = torch.tensor([i, j], device=scores.device)
            is_add[b] = channel_idx == 0
            rand_per_batch[b] = scores[b, channel_idx, i, j]
    
    return action_coords, is_add, rand_per_batch

@dataclass 
class BufferElement:
    S1_feats: torch.Tensor
    action: torch.Tensor
    action_idx: torch.Tensor
    reward: torch.Tensor
    S2_feats: torch.Tensor
    done: bool

# A replay buffer stores past experiences (state, action, reward, next_state, done), which are subsequently
# randomly sampled to break correlation between consecutive updates, which increases the stability of the network
class ReplayBuffer:
    def __init__(self, capacity: int = 400000):      # 4e5
        self.capacity = capacity
        self.buf = []
        self.pos = 0

    def __len__(self): return len(self.buf)

    # Add experience to the buffer
    def push(self, S1_feats, action, action_idx, reward, S2_feats, done) -> None:
        print("Current length of buffer: ", len(self.buf))
        print("Capacity of buffer: ", self.capacity)
        
        if (len(self.buf) + global_vars.batch_size) <= self.capacity:
            for b in range(global_vars.batch_size):
                buf_element = BufferElement(S1_feats[b], action[b], action_idx[b], reward[b], S2_feats[b], done)
                self.buf.append(buf_element)
        # Else, buffer is full, so we need to overwrite the oldest experience
        else:
            for b in range(global_vars.batch_size):
                self.buf[self.pos] = BufferElement(S1_feats[b], action[b], action_idx[b], reward[b], S2_feats[b], done)

        self.pos = (self.pos + global_vars.batch_size) % self.capacity

    # Sample a batch of experiences from the buffer
    def sample(self) -> List[BufferElement]:
        return random.sample(self.buf, global_vars.batch_size)
    
def compute_reward(current_metrics, next_metrics, w_area, w_delay, c_area: float = 1e-3, c_delay: float = 10.0) -> float:
    print("current_metrics: ", current_metrics)
    print("next_metrics: ", next_metrics)
    
    diff_area = current_metrics[:,1] - next_metrics[:,1]
    diff_delay = current_metrics[:,0] - next_metrics[:,0]
    
    return w_area*(c_area*diff_area) + w_delay*(c_delay*diff_delay)

# Training parameters from the PrefixRL paper
@dataclass
class TrainingConfig:
    gamma: float = 0.75                      # paper
    lr: float = 4e-5                         # paper
    target_sync_every: int = 60              # paper
    c_area: float = 1e-3                        # paper’s scaling
    c_delay: float = 10.0                       # paper’s scaling
    lr_decay: float = 0.99                    # exponential decay factor for learning rate
    
    
def train(cfg: TrainingConfig, device=None) -> Tuple[PrefixRL_DQN, PrefixRL_DQN]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # TODO: PrefixRL uses a double DQN architecture, where there is an online network (net) and an offline network (tgt)
    # Network actual being trained each step (also called online/policy network)
    net = PrefixRL_DQN(blocks=32, width=256).to(device)  # paper
    
    # Target network: slowly updated copy of the online network, providing stable targets for training
    tgt = PrefixRL_DQN(blocks=32, width=256).to(device)
    tgt.load_state_dict(net.state_dict()); tgt.eval()

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg.lr_decay)
    buf = ReplayBuffer(400000) # Initialize replay buffer with 4e5 elements

    grad_steps = 0

    # Weight scalar for area and delay
    w_area = global_vars.w_scalar
    w_delay = 1 - global_vars.w_scalar
    B = global_vars.batch_size
    
    # Main training loop
    for episode in range(global_vars.num_episodes):
        
        # Initial environment state at beginning of each episode
        current_state = init_graph(global_vars.n, global_vars.initial_adder_type)
        current_state.output_verilog()
        current_state.output_nodelist()
        current_state.run_yosys()
        delay,area,power = current_state.run_openroad()
        current_state_metrics = torch.tensor([delay, area, power]).repeat(B, 1)
        current_states = [current_state] * B
        
        # Sample a new value for epsilon-greedy exploration (PrefixRL Section III B)
        epsilon = sample_epsilon(episode)
        print("epsilon: ", epsilon)
        
        # Train for num_steps steps per episode
        for step in tqdm(range(global_vars.num_steps), desc="Training steps"):
            current_state_nodelist = torch.stack([torch.tensor(current_states[b].nodelist) for b in range(B)])
            current_state_minlist = torch.stack([torch.tensor(current_states[b].minlist) for b in range(B)])
            current_state_levellist = torch.stack([torch.tensor(current_states[b].levellist) for b in range(B)])
            current_state_fanoutlist = torch.stack([torch.tensor(current_states[b].fanoutlist) for b in range(B)])
            current_state_metrics = torch.stack([torch.tensor((current_states[b].delay, current_states[b].area, current_states[b].power)) for b in range(B)])
                
            feats = build_features(
                current_state_nodelist, 
                current_state_minlist, 
                current_state_levellist, 
                current_state_fanoutlist,
                batch_size=B
                ).to(device)   # (B,4,N,N) in [0,1]

            q = net(feats)
            
            # action_masks = build_action_masks(global_vars.n, current_state_nodelist, current_state_minlist)
            # q_masked = apply_action_masks(q, action_masks)
            
            q_masked = build_and_apply_action_masks_batch(q, feats, global_vars.n)
            q_scalarized = scalarize_q(q_masked, w_area, w_delay)
            
            # ε-greedy on scalarized add/del (choose random actions with probability ε)
            # If step == 0, choose a random action to start the episode
            rand_selection = torch.rand(1).item() < epsilon or step == 0
            if rand_selection:
                action_idx, action, value_per_batch = get_random_action(q_masked, w_area, w_delay)
            else:
                action_idx, action, value_per_batch = get_best_action(q_masked, w_area, w_delay)
                
            print("Shape of action_idx: ", action_idx.shape)
            print("action_idx: ", action_idx)
            print("Shape of action: ", action.shape)
            print("action: ", action)
            label = "rand" if rand_selection else "max"
            print(f"Shape of {label}_per_batch: {value_per_batch.shape}")
            print(f"{label}_per_batch: {value_per_batch}")
            
            # next_states is a list of B Graph_State objects
            # TODO: this will be the bottleneck of training. Should be performed in parallel instead of sequentially
            next_states = evaluate_next_state(current_states, action, action_idx[:,0], action_idx[:,1], B)
            next_state_nodelist = torch.stack([torch.tensor(next_states[b].nodelist) for b in range(B)])
            next_state_minlist = torch.stack([torch.tensor(next_states[b].minlist) for b in range(B)])
            next_state_levellist = torch.stack([torch.tensor(next_states[b].levellist) for b in range(B)])
            next_state_fanoutlist = torch.stack([torch.tensor(next_states[b].fanoutlist) for b in range(B)])
            next_state_metrics = torch.stack([torch.tensor((next_states[b].delay, next_states[b].area, next_states[b].power)) for b in range(B)])
            
            reward = compute_reward(current_state_metrics, next_state_metrics, w_area, w_delay)
            
            # Build features for the next state (inputs are already batched)
            next_feats = build_features(
                next_state_nodelist, 
                next_state_minlist, 
                next_state_levellist, 
                next_state_fanoutlist
                ).to(device)   # (B,4,N,N) in [0,1]
            
            q_next = net(next_feats)
            
            q_next_masked = build_and_apply_action_masks_batch(q_next, next_feats, global_vars.n)
            q_next_scalarized = scalarize_q(q_next_masked, w_area, w_delay)
            
            # print("Shape of feats: ", feats.shape)
            # print("feats: ", feats)
            # print("Shape of next_feats: ", next_feats.shape)
            # print("next_feats: ", next_feats)

            # Push the current state, taken action, node index, reward, and next_state to the replay buffer
            buf.push(feats.detach(), action.detach(), action_idx.detach(), reward.detach(), next_feats.detach(), False)
            
            # Sample experience buffer
            sampled_elements = buf.sample()  # shapes (B,...)
            print("Shape of sampled_elements: ", len(sampled_elements))
            # print("sampled_elements: ", sampled_elements)
            
            # Q(s) from online network - needs gradients
            replay_S1_feats = torch.stack([elem.S1_feats for elem in sampled_elements])
            replay_action = torch.stack([elem.action for elem in sampled_elements])
            replay_action_idx = torch.stack([elem.action_idx for elem in sampled_elements])
            replay_reward = torch.stack([elem.reward for elem in sampled_elements])
            replay_S2_feats = torch.stack([elem.S2_feats for elem in sampled_elements])
            replay_done = torch.stack([torch.tensor(elem.done, dtype=torch.float32) for elem in sampled_elements])
            
            # Recompute Q(S1) from online network (with gradients)
            Q_S1 = net(replay_S1_feats)
            q_S1_masked = build_and_apply_action_masks_batch(Q_S1, replay_S1_feats, global_vars.n)
            Q_S1_scalarized = scalarize_q(q_S1_masked, w_area, w_delay)
            
            B = Q_S1_scalarized.size(0)
            batch_idx = torch.arange(B, device=Q_S1_scalarized.device)
            replay_action_type = (~replay_action).long()  # 0 for add, 1 for delete
            
            replay_q_sa = Q_S1_scalarized[batch_idx, replay_action_type, replay_action_idx[:, 0], replay_action_idx[:, 1]]
            
            # Q(s',a') from target network - no gradients needed
            with torch.no_grad():
                Q_S2 = tgt(replay_S2_feats)
                q_S2_masked = build_and_apply_action_masks_batch(Q_S2, replay_S2_feats, global_vars.n)
                action_coords, is_add, replay_q_max_per_batch = get_best_action(q_S2_masked, w_area, w_delay)
                target = replay_reward + cfg.gamma * replay_q_max_per_batch
                
            loss = F.smooth_l1_loss(replay_q_sa, target)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            grad_steps += 1
            
            # Sync the target network with the online network every 60 steps
            if grad_steps % cfg.target_sync_every == 0:       # every 60 steps
                tgt.load_state_dict(net.state_dict())
                
            current_states = next_states
            current_state_metrics = next_state_metrics
            
            training_log_entry = ""
            for b in range(B):
                training_log_entry += "{},{},{},{},{},{},{},{}\n".format(
                    time.time(),
                    episode,
                    step,
                    replay_reward[b].item(),
                    target[b].item(),
                    replay_q_sa[b].item(),
                    replay_q_max_per_batch[b].item(),
                    loss.item()
                )
            global_vars.training_log.write(training_log_entry)
            global_vars.training_log.flush()

        scheduler.step()

    return net, tgt
    
    
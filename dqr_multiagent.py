# Necessary imports
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import pickle
import matplotlib.pyplot as plt
import copy # For potentially creating a lagging network if desired later
import traci

# This file now implements the Attentive Q-Mixing (A-QMIX) architecture.


from hub_routing_env import HubRoutingEnv

# Set seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 0.1 Attention Module for A-QMIX ---
class AttentionModule(nn.Module):
    """
    Attention module for aggregating multiple decision utilities from a single agent
    during a Global Collection Epoch (GCE).
    """
    def __init__(self, global_state_dim, local_state_dim, hidden_dim=64): # MODIFIED
        """
        Args:
            global_state_dim (int): Dimension of the global state vector (for Query).
            local_state_dim (int): Dimension of the local state vector (for Keys).
            hidden_dim (int): Hidden dimension for MLP networks.
        """
        super(AttentionModule, self).__init__()
        
        self.key_network = nn.Sequential(
            nn.Linear(local_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.query_network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.apply(self._init_weights)
        print(f"Initialized AttentionModule: global_state_dim={global_state_dim}, local_state_dim={local_state_dim}, hidden_dim={hidden_dim}")
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
                
    def forward(self, global_state, local_states_for_decisions, decision_utilities):
        """
        Forward pass: Computes attention-weighted aggregate of decision utilities.
        
        Args:
            global_state (torch.Tensor): Global state tensor [batch_size_gce, global_state_dim]
                                         (Note: often batch_size_gce will be 1 if called per GCE item)
            local_states_for_decisions (torch.Tensor): Local state tensors for each decision 
                                                       [num_decisions_in_gce, local_state_dim]
            decision_utilities (torch.Tensor): Q-values (utilities) for each decision 
                                               [num_decisions_in_gce, 1]
            
        Returns:
            torch.Tensor: Attention-weighted aggregate utility [batch_size_gce, 1]
        """
        if local_states_for_decisions.size(0) == 0:
            return torch.zeros(global_state.size(0), 1, device=global_state.device)
        
        keys = self.key_network(local_states_for_decisions)  # [num_decisions, hidden_dim//2]
        query = self.query_network(global_state)             # [batch_size_gce, hidden_dim//2]
        
        query_expanded = query.unsqueeze(1)  # [batch_size_gce, 1, hidden_dim//2]
        keys_expanded = keys.unsqueeze(0)    # [1, num_decisions, hidden_dim//2]
        
        scale_factor = (keys.size(-1)) ** 0.5
        attention_scores = torch.bmm(query_expanded, keys_expanded.transpose(1, 2)) / scale_factor # [batch_size_gce, 1, num_decisions]
        
        attention_weights = F.softmax(attention_scores, dim=2) # [batch_size_gce, 1, num_decisions]
        
        utilities_expanded = decision_utilities.unsqueeze(0).transpose(1,2)  # [1, 1, num_decisions] to match weights
        
        # Corrected broadcasting for utilities
        # attention_weights is [batch_size_gce, 1, num_decisions]
        # decision_utilities is [num_decisions, 1], needs to be [1, num_decisions, 1] for bmm or element-wise
        weighted_utilities = torch.bmm(attention_weights, decision_utilities.unsqueeze(0)) # [batch_size_gce, 1, 1]
        
        attended_utility = weighted_utilities.squeeze(2) # [batch_size_gce, 1]
        
        return attended_utility

# --- 0.2 QMIX Mixing Network ---
class QMixingNetwork(nn.Module):
    def __init__(self, state_dim, num_agents, hidden_dim=64):
        super(QMixingNetwork, self).__init__()
        self.state_dim = state_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim)
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(self._init_weights)
        print(f"Initialized QMixingNetwork: state_dim={state_dim}, num_agents={num_agents}, hidden_dim={hidden_dim}")
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
                
    def forward(self, global_state, agent_attended_utilities):
        batch_size = global_state.size(0)
        
        w1 = torch.abs(self.hyper_w1(global_state))
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)
        b1 = self.hyper_b1(global_state)
        
        hidden = F.elu(torch.bmm(agent_attended_utilities.unsqueeze(1), w1).squeeze(1) + b1)
        
        w2 = torch.abs(self.hyper_w2(global_state))
        b2 = self.hyper_b2(global_state)
        
        q_total = torch.bmm(hidden.unsqueeze(1), w2.unsqueeze(2)).squeeze(1) + b2
        return q_total

# --- 1. Time Estimation Network (Agent's Q_a network) ---
class TimeEstimationNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(TimeEstimationNetwork, self).__init__()
        self.action_dim = action_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.apply(self._init_weights)
        print(f"Initialized TimeEstimationNetwork: state_dim={state_dim}, action_dim={action_dim}, hidden_dim={hidden_dim}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)

    def forward(self, state_data):
        state = state_data['state']
        apply_bn = state.size(0) > 1
        x = self.fc1(state)
        if apply_bn: x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if apply_bn: x = self.bn2(x)
        x = F.relu(x)
        time_estimates = self.fc3(x)
        return time_estimates

# --- 2. Replay Buffers ---
Experience = namedtuple("Experience",
                        field_names=["state", "action", "hop_time", "next_state",
                                     "destination_hub_id", "neighbor_hub_id", "next_state_action_mask", "done"])

class QRouterReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        # ... (rest of QRouterReplayBuffer remains the same as provided)
        self.capacity = capacity
        print(f"Initialized Replay Buffer with capacity {capacity}")

    def add(self, state, action, hop_time, next_state, destination_hub_id, neighbor_hub_id, next_state_action_mask, done):
        if state is None or next_state is None:
             #print(f"Warning: Attempted to add experience with None state/next_state. Skipping.")
             return
        state = np.array(state, dtype=np.float32)
        action = int(action)
        hop_time = float(hop_time)
        next_state = np.array(next_state, dtype=np.float32)
        destination_hub_id = str(destination_hub_id)
        neighbor_hub_id = str(neighbor_hub_id)
        if next_state_action_mask is not None:
            next_state_action_mask = np.array(next_state_action_mask, dtype=bool)
        done = bool(done)
        experience = Experience(state, action, hop_time, next_state, destination_hub_id, neighbor_hub_id, next_state_action_mask, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        experiences = random.sample(self.buffer, batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        hop_times = torch.from_numpy(np.vstack([e.hop_time for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        destination_hub_ids = [e.destination_hub_id for e in experiences]
        neighbor_hub_ids = [e.neighbor_hub_id for e in experiences]
        next_state_action_masks = []
        for e in experiences:
            if e.next_state_action_mask is None:
                next_state_action_masks.append(None)
            else:
                 next_state_action_masks.append(torch.from_numpy(e.next_state_action_mask).bool().to(device))
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        return (states, actions, hop_times, next_states, destination_hub_ids,
                neighbor_hub_ids, next_state_action_masks, dones)

    def __len__(self):
        return len(self.buffer)

class GlobalCollectionEpochBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        print(f"Initialized GCE Buffer with capacity {capacity}")
        
    def add(self, global_state, agent_decisions, reward, next_global_state, next_local_states_all, next_local_states_action_masks_all, done): # MODIFIED: Added next_local_states_action_masks_all
        global_state = np.array(global_state, dtype=np.float32)
        next_global_state = np.array(next_global_state, dtype=np.float32)
        reward = float(reward)
        done = bool(done)
        
        agent_decisions_copy = {}
        for agent_id, decisions in agent_decisions.items():
            agent_decisions_copy[agent_id] = []
            for decision in decisions:
                decision_copy = {'state': np.array(decision['state'], dtype=np.float32) if decision['state'] is not None else None,
                                 'action': decision['action']}
                agent_decisions_copy[agent_id].append(decision_copy)

        next_local_states_all_copy = {}
        if next_local_states_all is not None:
            for agent_id, state in next_local_states_all.items():
                next_local_states_all_copy[agent_id] = np.array(state, dtype=np.float32) if state is not None else None
        
        # ADDED: Store next state action masks
        next_local_states_action_masks_all_copy = {}
        if next_local_states_action_masks_all is not None:
            for agent_id, action_mask in next_local_states_action_masks_all.items():
                next_local_states_action_masks_all_copy[agent_id] = np.array(action_mask, dtype=bool) if action_mask is not None else None
        
        gce_transition = {
            'global_state': global_state,
            'agent_decisions': agent_decisions_copy, # Dict[agent_id, List[Dict{'state', 'action'}]]
            'reward': reward,
            'next_global_state': next_global_state,
            'next_local_states_all': next_local_states_all_copy, # Dict[agent_id, local_state_np_array] MODIFIED
            'next_local_states_action_masks_all': next_local_states_action_masks_all_copy, # ADDED
            'done': done
        }
        self.buffer.append(gce_transition)
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        
        transitions = random.sample(self.buffer, batch_size)
        
        global_states = torch.from_numpy(np.vstack([t['global_state'] for t in transitions])).float().to(device)
        rewards = torch.from_numpy(np.vstack([t['reward'] for t in transitions])).float().to(device)
        next_global_states = torch.from_numpy(np.vstack([t['next_global_state'] for t in transitions])).float().to(device)
        dones = torch.from_numpy(np.vstack([t['done'] for t in transitions]).astype(np.uint8)).float().to(device)
        
        agent_decisions_batch = [t['agent_decisions'] for t in transitions]
        next_local_states_all_batch = [t['next_local_states_all'] for t in transitions] # MODIFIED
        next_local_states_action_masks_all_batch = [t['next_local_states_action_masks_all'] for t in transitions] # ADDED
        
        return global_states, agent_decisions_batch, rewards, next_global_states, next_local_states_all_batch, next_local_states_action_masks_all_batch, dones # MODIFIED
    
    def __len__(self):
        return len(self.buffer)

# --- 3. Hub Q-Router Agent (A-QMIX version) ---
class HubQRouterAgent:
    def __init__(self, hub_id, local_state_dim, action_dim, global_state_dim, # MODIFIED: added global_state_dim
                 lr=1e-4, gamma=1.0,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.999): # REMOVED buffer_capacity

        self.hub_id = hub_id
        self.local_state_dim = local_state_dim # MODIFIED: renamed state_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim # MODIFIED
        self.gamma = gamma # Retained for potential future use, but not in A-QMIX target directly
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.time_network = TimeEstimationNetwork(local_state_dim, action_dim).to(device) # MODIFIED
        self.target_time_network = TimeEstimationNetwork(local_state_dim, action_dim).to(device) # MODIFIED
        self.target_time_network.load_state_dict(self.time_network.state_dict())
        self.target_time_network.eval()

        self.attention_module = AttentionModule(global_state_dim, local_state_dim).to(device) # MODIFIED
        self.target_attention_module = AttentionModule(global_state_dim, local_state_dim).to(device) # MODIFIED
        self.target_attention_module.load_state_dict(self.attention_module.state_dict())
        self.target_attention_module.eval()

        self.optimizer = optim.AdamW(
            list(self.time_network.parameters()) + list(self.attention_module.parameters()),
            lr=lr, amsgrad=True
        )
        
        # REMOVED: self.replay_buffer - Individual agent buffers not used for A-QMIX training logic directly

        self.gce_decisions = [] 
        self.update_steps = 0 # Kept for potential agent-specific tracking if needed later
        
        print(f"Initialized HubQRouterAgent for hub '{self.hub_id}': local_state_dim={self.local_state_dim}, action_dim={self.action_dim}, global_state_dim={self.global_state_dim}")


    def select_action(self, state, action_mask):
        # ... (select_action method remains the same as provided previously)
        eps = self.epsilon
        if random.random() < eps:
            if action_mask is not None:
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0: return np.random.choice(valid_actions)
                else: return random.randint(0, self.action_dim - 1)
            else: return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                self.time_network.eval()
                state_data = {'state': state_tensor}
                time_estimates = self.time_network(state_data)[0]
                self.time_network.train()
                if action_mask is not None:
                    mask_tensor = torch.from_numpy(action_mask).bool().to(device)
                    time_estimates = time_estimates.to(device) # Ensure same device
                    time_estimates[~mask_tensor] = float('inf')
                best_action = torch.argmin(time_estimates).item()
                if torch.isinf(time_estimates.min()):
                    valid_actions = np.where(action_mask)[0] if action_mask is not None else []
                    if len(valid_actions) > 0: return np.random.choice(valid_actions)
                    else: return random.randint(0, self.action_dim - 1)
                return best_action

    def record_decision(self, state, action): # MODIFIED: removed reward
        self.gce_decisions.append({'state': state, 'action': action})
        
    def clear_gce_decisions(self):
        self.gce_decisions = []

    def get_attended_utility(self, global_state, decisions_for_this_gce, target=False): # MODIFIED: decisions_for_this_gce passed in
        if not decisions_for_this_gce:
            return torch.zeros(global_state.size(0), 1, device=device) # global_state batch_size respected
        
        # This method assumes global_state is [1, global_state_dim] if called per GCE item,
        # or [batch_gce_size, global_state_dim] if vectorized.
        # For now, let's assume it's called per GCE item (batch_size_gce=1 for this call).
        # If global_state is batched, and we process one agent's GCE decisions,
        # this implies decisions_for_this_gce is for one GCE, and global_state for that GCE.

        with torch.set_grad_enabled(not target):
            states = np.vstack([d['state'] for d in decisions_for_this_gce])
            actions = np.array([d['action'] for d in decisions_for_this_gce])
            
            states_tensor = torch.from_numpy(states).float().to(device)
            actions_tensor = torch.from_numpy(actions).long().to(device)
            
            state_data = {'state': states_tensor}
            
            q_network_to_use = self.target_time_network if target else self.time_network
            attention_module_to_use = self.target_attention_module if target else self.attention_module
            
            if target:
                q_network_to_use.eval()
                attention_module_to_use.eval()
            else: # ensure online nets are in train mode if not target
                q_network_to_use.train() 
                attention_module_to_use.train()

            all_q_values = q_network_to_use(state_data) # [num_decisions, action_dim]
            q_values_for_taken_actions = all_q_values.gather(1, actions_tensor.unsqueeze(1)) # [num_decisions, 1]
            
            # Ensure global_state matches expected input for attention if it's for a single GCE item
            # If global_state passed in is already [1, global_state_dim], it's fine.
            attended_utility = attention_module_to_use(global_state, states_tensor, q_values_for_taken_actions)
            
            return attended_utility # Should be [1, 1] or [batch_size_gce, 1]

    def compute_target_q_values_for_state(self, next_local_state, next_action_mask=None): # NEW METHOD
        """Computes Q'_a(s'_a, argmax_u'' Q_a(s'_a, u'')) for a given next_local_state."""
        if next_local_state is None:
            # Handle cases where a next local state might not be applicable (e.g. terminal)
            # or an agent wasn't 'active' to have a defined next_local_state from the GCE perspective
             return torch.tensor([[0.0]], device=device) # Default to zero utility


        next_state_tensor = torch.from_numpy(next_local_state).float().unsqueeze(0).to(device)
        next_state_data = {'state': next_state_tensor}

        with torch.no_grad():
            self.time_network.eval() # Use online network for argmax
            self.target_time_network.eval()

            online_q_values_next = self.time_network(next_state_data)[0] # [action_dim]
            
            if next_action_mask is not None: # Assuming next_action_mask is a numpy array
                mask_tensor = torch.from_numpy(next_action_mask).bool().to(device)
                online_q_values_next[~mask_tensor] = float('inf') # Lower is better for time

            best_next_action = torch.argmin(online_q_values_next).unsqueeze(0) # [1]

            if torch.isinf(online_q_values_next.min()): # All actions masked
                target_q_value = torch.tensor([[1000.0]], device=device) # Large penalty
            else:
                target_q_values_next_all_actions = self.target_time_network(next_state_data) # [1, action_dim]
                target_q_value = target_q_values_next_all_actions.gather(1, best_next_action.unsqueeze(0)) # [1,1]
        
        self.time_network.train() # Set back to train mode
        return target_q_value


    def soft_update_target_network(self, polyak=0.995):
        with torch.no_grad():
            for target_param, online_param in zip(self.target_time_network.parameters(), self.time_network.parameters()):
                target_param.data.copy_(polyak * target_param.data + (1.0 - polyak) * online_param.data)
            for target_param, online_param in zip(self.target_attention_module.parameters(), self.attention_module.parameters()):
                target_param.data.copy_(polyak * target_param.data + (1.0 - polyak) * online_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        # ... (save method remains similar, ensure all relevant components are saved)
        torch.save({
            'time_network_state_dict': self.time_network.state_dict(),
            'target_time_network_state_dict': self.target_time_network.state_dict(),
            'attention_module_state_dict': self.attention_module.state_dict(),
            'target_attention_module_state_dict': self.target_attention_module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath):
        # ... (load method remains similar, ensure all relevant components are loaded)
        if not os.path.exists(filepath):
            print(f"Warning: Checkpoint file not found at {filepath}. Agent not loaded.")
            return
        checkpoint = torch.load(filepath, map_location=device)
        self.time_network.load_state_dict(checkpoint['time_network_state_dict'])
        self.target_time_network.load_state_dict(checkpoint.get('target_time_network_state_dict', self.time_network.state_dict()))
        self.attention_module.load_state_dict(checkpoint['attention_module_state_dict'])
        self.target_attention_module.load_state_dict(checkpoint.get('target_attention_module_state_dict', self.attention_module.state_dict()))
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.time_network.to(device)
        self.target_time_network.to(device)
        self.target_time_network.eval()
        self.attention_module.to(device)
        self.target_attention_module.to(device)
        self.target_attention_module.eval()
        print(f"Loaded agent state for hub '{self.hub_id}' from {filepath}")


# --- 4. AQMIX Trainer ---
class AQMIXTrainer: # MODIFIED: No longer inherits from MultiAgentQRouterTrainer to avoid confusion
    def __init__(self, env, config):
        self.env = env
        self.config = config

        # NEW: local_state_dim = z_order_embedding + current_hub_conditions + (num_neighbor_features * max_neighbors)
        self.local_state_dim = env.z_order_embedding_dim + 2 + (env.max_neighbors * 3)
        
        # NEW: Calculate global state dimension based on the new spec
        num_hubs_from_env = len(env.hub_agents)
        hub_centric_dim = 2 * num_hubs_from_env      # vicinity_speed + processing_rate for each hub
        network_wide_dim = 3                         # avg_inefficiency + demand_supply + vehicle_count
        system_imbalance_dim = 1                     # std_dev of vicinity_speeds
        self.global_state_dim = hub_centric_dim + network_wide_dim + system_imbalance_dim
        
        print(f"Calculated local_state_dim: {self.local_state_dim}, global_state_dim: {self.global_state_dim}")

        self.agents = {}
        for hub_id, env_hub_agent_info in env.hub_agents.items():
            action_dim = env_hub_agent_info.action_space_size
            if action_dim <= 0:
                 print(f"Warning: Hub '{hub_id}' has action_dim <= 0. Skipping agent creation.")
                 continue
            self.agents[hub_id] = HubQRouterAgent(
                hub_id=hub_id,
                local_state_dim=self.local_state_dim, # MODIFIED
                action_dim=action_dim,
                global_state_dim=self.global_state_dim, # MODIFIED
                lr=config.get('lr', 1e-4),
                gamma=config.get('gamma', 1.0),
                epsilon_start=config.get('epsilon_start', 1.0),
                epsilon_end=config.get('epsilon_end', 0.05),
                epsilon_decay=config.get('epsilon_decay', 0.9995)
                # buffer_capacity removed as it's not used by agent in A-QMIX
            )
        if not self.agents: raise RuntimeError("No agents created.")

        self.mixing_network = QMixingNetwork(self.global_state_dim, len(self.agents), config.get('mixing_hidden_dim', 64)).to(device)
        self.target_mixing_network = QMixingNetwork(self.global_state_dim, len(self.agents), config.get('mixing_hidden_dim', 64)).to(device)
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        self.target_mixing_network.eval()
        
        self.gce_buffer = GlobalCollectionEpochBuffer(config.get('gce_buffer_capacity', 10000))
        
        self.mixing_optimizer = optim.AdamW(self.mixing_network.parameters(), lr=config.get('mixing_lr', 1e-4), amsgrad=True)
        
        self.current_gce = {} # To be initialized by _init_new_gce
        self.gce_size = config.get('gce_size', len(self.agents) * 3)
        self.gce_max_sim_time = config.get('gce_max_sim_time', 100)
        
        self.total_steps = 0
        self.episode_actual_travel_times = []
        self.episode_completion_rates = []
        self.episode_steps = []
        self.mixing_losses = [] # For QMIX mixing network
        self.agent_specific_losses_proxy = {hub_id: [] for hub_id in self.agents} # For tracking magnitude of updates if desired

        print(f"Initialized AQMIXTrainer with {len(self.agents)} agents.")

    def log_global_state(self, global_state, step_info=""):
        """Logs the components of the global state vector with labels."""
        print(f"\n--- Logging Global State {step_info} ---")
        if global_state is None or global_state.size == 0:
            print("Global state is empty or None.")
            return

        num_hubs = len(self.agents)
        hub_centric_dim = 2 * num_hubs
        network_wide_dim = 3
        system_imbalance_dim = 1
        
        expected_dim = hub_centric_dim + network_wide_dim + system_imbalance_dim
        if global_state.shape[0] != expected_dim:
            print(f"ERROR: Global state dimension mismatch. Expected {expected_dim}, got {global_state.shape[0]}")
            return

        idx = 0
        print("  1. Hub-Centric Conditions:")
        for hub_id in sorted(self.agents.keys()):
            print(f"     - Hub '{hub_id}' Vicinity Speed: {global_state[idx]:.4f}")
            idx += 1
            print(f"     - Hub '{hub_id}' Processing Rate: {global_state[idx]:.4f}")
            idx += 1

        print("  2. Network-Wide Metrics:")
        print(f"     - Avg Trip Inefficiency:    {global_state[idx]:.4f}")
        idx += 1
        print(f"     - Demand/Supply Balance:    {global_state[idx]:.4f}")
        idx += 1
        print(f"     - Network Vehicle Count:    {global_state[idx]:.4f}")
        idx += 1
        
        print("  3. System Imbalance:")
        print(f"     - Std Dev of Vicinity Speeds: {global_state[idx]:.4f}")
        
        print("--- End Global State Log ---")

    def _get_global_state_and_local_states(self):
        """
        Constructs the new flow-based global state vector for the QMIX network.
        
        Global State Vector Components:
        - Per-Hub Flow Snapshot (Real-time Bottleneck Map):
          For each hub: [hub_vicinity_speed_normalized, hub_outgoing_congestion_ratio]
        - System-Wide Load & Efficiency:
          - total_vehicle_count_normalized
          - completion_throughput_ratio  
          - avg_trip_inefficiency_ratio
        - System Imbalance Metric (Flow-based):
          - std_dev_of_vicinity_speeds
        """
        # 1. Per-Hub Flow Snapshot (Real-time Bottleneck Map)
        hub_flow_features = []
        hub_vicinity_speeds = []

        for hub_id in sorted(self.agents.keys()):
            # A. Hub Vicinity Speed Normalized
            vicinity_speed = self.env.get_hub_vicinity_speed(hub_id)
            hub_vicinity_speeds.append(vicinity_speed)
            
            # B. Hub Outgoing Congestion Ratio  
            outgoing_congestion = self.env.get_hub_outgoing_congestion_ratio(hub_id)
            
            hub_flow_features.extend([vicinity_speed, outgoing_congestion])

        # 2. System-Wide Load & Efficiency
        # A. Total Vehicle Count Normalized
        try:
            num_vehicles = len(traci.vehicle.getIDList())
        except Exception:  # Handle if traci is not available
            num_vehicles = 0
        total_vehicle_count_normalized = min(1.0, num_vehicles / self.env.max_vehicles)
        
        # B. Completion Throughput Ratio (System efficiency: completed / started)
        completion_throughput_ratio = self.env.get_completion_throughput_ratio()
        
        # C. Average Trip Inefficiency Ratio (Routing quality)
        avg_inefficiency = self.env.get_avg_completed_trip_inefficiency()
        # Normalize inefficiency: 1.0 is perfect, higher is worse. Cap at 5x for normalization.
        avg_trip_inefficiency_ratio = min(1.0, (avg_inefficiency - 1.0) / 4.0)

        system_wide_features = [total_vehicle_count_normalized, completion_throughput_ratio, avg_trip_inefficiency_ratio]

        # 3. System Imbalance Metric (Flow-based)
        # Standard deviation of vicinity speeds across all hubs indicates flow imbalance
        std_dev_speeds = np.std(hub_vicinity_speeds) if hub_vicinity_speeds else 0.0
        # Normalize std dev: 0 is balanced, higher is imbalanced. Assume max std dev of 0.5 for normalization.
        std_dev_of_vicinity_speeds = min(1.0, std_dev_speeds / 0.5)
        
        imbalance_feature = [std_dev_of_vicinity_speeds]

        # 4. Concatenate all features into the final global state
        global_state_np = np.concatenate([hub_flow_features, system_wide_features, imbalance_feature]).astype(np.float32)

        # The local states part is required for the buffer, but its logic can remain simple as before.
        # It provides the *next* local states for the agents at the end of a GCE.
        current_local_states = {}
        current_local_states_action_masks = {}  # ADDED: Store action masks for next states
        
        for hub_id in self.agents.keys():
            representative_local_state_for_hub = np.zeros(self.local_state_dim, dtype=np.float32)
            representative_action_mask_for_hub = None  # ADDED
            
            vehicles_at_this_hub = [v_id for v_id, v_data in self.env.vehicles.items() 
                                   if v_data.get('current_hub') == hub_id and v_data.get('waiting_at_hub', False)]
            if vehicles_at_this_hub:
                # Use the state for the first waiting vehicle as representative
                state_data_tuple = self.env._get_dqr_state_for_vehicle(hub_id, vehicles_at_this_hub[0])
                if state_data_tuple:
                    representative_local_state_for_hub, representative_action_mask_for_hub = state_data_tuple  # MODIFIED: Unpack both state and action mask
            
            current_local_states[hub_id] = representative_local_state_for_hub
            current_local_states_action_masks[hub_id] = representative_action_mask_for_hub  # ADDED

        return global_state_np, current_local_states, current_local_states_action_masks  # MODIFIED: Return action masks too


    def _init_new_gce(self):
        g_state, _, _ = self._get_global_state_and_local_states() # We only need S_glob here, updated to handle 3 return values
        self.current_gce = {
            'global_state': g_state,
            'agent_decisions': {agent_id: [] for agent_id in self.agents}, # Will store lists of {'state', 'action'}
            'resolved_rewards': [], # List of scalar rewards
            'initiated_decisions_count': 0,
            'start_time': self.env.current_time,
            'next_local_states_all': {agent_id: None for agent_id in self.agents}, # To be filled at GCE end
            'next_local_states_action_masks_all': {agent_id: None for agent_id in self.agents} # To be filled at GCE end
        }
        for agent in self.agents.values(): agent.clear_gce_decisions()

    def _end_current_gce(self, terminated=False):
        next_g_state, next_l_states_all, next_l_states_action_masks_all = self._get_global_state_and_local_states() # Capture S'_glob, s'_a, and s'_a action masks

        self.current_gce['next_global_state'] = next_g_state
        self.current_gce['next_local_states_all'] = next_l_states_all # Store s'_a for all agents
        self.current_gce['next_local_states_action_masks_all'] = next_l_states_action_masks_all # Store s'_a action masks for all agents

        total_reward_for_gce = sum(self.current_gce['resolved_rewards'])
        gce_duration = self.env.current_time - self.current_gce['start_time']
        if not terminated and \
        self.current_gce['initiated_decisions_count'] < self.gce_size and \
        gce_duration > self.gce_max_sim_time: # If timed out (with a small buffer)
            timeout_penalty = 0
            total_reward_for_gce += timeout_penalty
        
        # agent_decisions for buffer should be a dict: {agent_id: list_of_decision_dicts}
        # where each decision_dict is {'state': local_state_np, 'action': action_int}
        # HubQRouterAgent.gce_decisions already stores this.
        agent_decisions_to_store = {agent_id: agent.gce_decisions.copy() for agent_id, agent in self.agents.items()}

        self.gce_buffer.add(
            global_state=self.current_gce['global_state'],
            agent_decisions=agent_decisions_to_store,
            reward=total_reward_for_gce,
            next_global_state=self.current_gce['next_global_state'],
            next_local_states_all=self.current_gce['next_local_states_all'], # MODIFIED
            next_local_states_action_masks_all=self.current_gce['next_local_states_action_masks_all'], # MODIFIED
            done=terminated
        )
        self._init_new_gce() # Start a new GCE
    
    def _should_end_gce(self, terminated):
        # ... (remains the same)
        if terminated: return True
        if self.current_gce['initiated_decisions_count'] >= self.gce_size: return True
        current_sim_time = self.env.current_time
        gce_start_time = self.current_gce['start_time']
        if current_sim_time - gce_start_time > self.gce_max_sim_time: return True
        return False
    
    def _update_qmix(self, batch_size):
        batch = self.gce_buffer.sample(batch_size)
        if batch is None: return None
            
        global_states_b, agent_decisions_b, rewards_b, next_global_states_b, next_local_states_all_b, next_local_states_action_masks_all_b, dones_b = batch
        # global_states_b is [batch_size, global_state_dim]
        # agent_decisions_b is a list of length batch_size, each element is a dict {agent_id: list_of_decisions}
        # rewards_b is [batch_size, 1]
        # next_global_states_b is [batch_size, global_state_dim]
        # next_local_states_all_b is a list of length batch_size, each element is {agent_id: s'_a}
        # next_local_states_action_masks_all_b is a list of length batch_size, each element is {agent_id: s'_a action mask}
        # dones_b is [batch_size, 1]

        current_attended_utilities_list = []
        target_q_prime_utilities_list = []

        for b_idx in range(batch_size):
            # For Q_tot calculation
            gce_agent_decisions = agent_decisions_b[b_idx] # Decisions for GCE item 'b_idx'
            current_s_glob_item = global_states_b[b_idx:b_idx+1] # [1, global_state_dim]
            
            # For Q'_tot calculation
            next_s_glob_item = next_global_states_b[b_idx:b_idx+1] # [1, global_state_dim]
            next_s_prime_locals_item = next_local_states_all_b[b_idx] # dict {agent_id: s'_a}
            next_s_prime_locals_action_masks_item = next_local_states_action_masks_all_b[b_idx] # dict {agent_id: s'_a action mask}

            attended_utilities_for_gce_item = []
            q_prime_utilities_for_gce_item = []

            for agent_id in sorted(self.agents.keys()): # Consistent agent order
                agent = self.agents[agent_id]
                
                # --- Calculate Q_a_star for current Q_tot ---
                decisions_this_agent_this_gce = gce_agent_decisions.get(agent_id, [])
                q_star_a = agent.get_attended_utility(
                    global_state=current_s_glob_item,
                    decisions_for_this_gce=decisions_this_agent_this_gce, # Pass the specific decisions
                    target=False
                ) # Output shape [1, 1]
                attended_utilities_for_gce_item.append(q_star_a)

                # --- Calculate Q'_a for target Q'_tot ---
                s_prime_a_np = next_s_prime_locals_item.get(agent_id)
                s_prime_a_action_mask_np = next_s_prime_locals_action_masks_item.get(agent_id)
                q_prime_a = agent.compute_target_q_values_for_state(s_prime_a_np, next_action_mask=s_prime_a_action_mask_np) # Output shape [1,1]
                q_prime_utilities_for_gce_item.append(q_prime_a)

            current_attended_utilities_list.append(torch.cat(attended_utilities_for_gce_item, dim=1)) # Shape [1, num_agents]
            target_q_prime_utilities_list.append(torch.cat(q_prime_utilities_for_gce_item, dim=1))   # Shape [1, num_agents]

        # Stack to create batch tensors
        batch_attended_utilities = torch.cat(current_attended_utilities_list, dim=0) # [batch_size, num_agents]
        batch_target_q_prime_utilities = torch.cat(target_q_prime_utilities_list, dim=0) # [batch_size, num_agents]

        # Compute Q_tot
        self.mixing_network.train()
        q_tot = self.mixing_network(global_states_b, batch_attended_utilities) # [batch_size, 1]
        
        # Compute target Q'_tot
        with torch.no_grad():
            self.target_mixing_network.eval()
            target_q_tot = self.target_mixing_network(next_global_states_b, batch_target_q_prime_utilities) # [batch_size, 1]
            
        target_y = rewards_b + self.config.get('gamma', 1.0) * (1 - dones_b) * target_q_tot
        
        loss = F.smooth_l1_loss(q_tot, target_y.detach()) # Detach target
        
        # Zero gradients for all optimizers
        self.mixing_optimizer.zero_grad()
        for agent in self.agents.values():
            agent.optimizer.zero_grad()
            
        loss.backward()
        
        # Clip gradients and step optimizers
        torch.nn.utils.clip_grad_norm_(self.mixing_network.parameters(), max_norm=self.config.get('clip_grad_norm', 1.0))
        self.mixing_optimizer.step()
        
        for agent in self.agents.values():
            torch.nn.utils.clip_grad_norm_(agent.time_network.parameters(), max_norm=self.config.get('clip_grad_norm', 1.0))
            torch.nn.utils.clip_grad_norm_(agent.attention_module.parameters(), max_norm=self.config.get('clip_grad_norm', 1.0))
            agent.optimizer.step()
        
        loss_value = loss.item()
        self.mixing_losses.append(loss_value)
        return loss_value
    
    def _soft_update_mixing_network(self, polyak=0.995):
        with torch.no_grad():
            for target_param, online_param in zip(self.target_mixing_network.parameters(), self.mixing_network.parameters()):
                target_param.data.copy_(polyak * target_param.data + (1.0 - polyak) * online_param.data)
    
    def train(self, num_episodes):
        # ... (Train loop structure remains similar to AQMIXTrainer in provided code)
        # Key differences:
        # 1. No individual agent.update() call.
        # 2. agent.record_decision only takes state and action.
        # 3. Logic to get s_prime_a for all agents at end of GCE for buffer.

        min_gce_buffer_size = self.config.get('min_gce_buffer_size', 100)
        qmix_batch_size = self.config.get('qmix_batch_size', 32)
        qmix_update_frequency = self.config.get('qmix_update_frequency_steps', 8) # Changed from steps to GCEs or steps
        save_frequency = self.config.get('save_frequency_episodes', 50)
        log_frequency = self.config.get('log_frequency_episodes', 10)
        save_dir = self.config.get('save_dir', 'aqmix_models')
        polyak = self.config.get('polyak', 0.995)

        os.makedirs(save_dir, exist_ok=True)
        print(f"\n--- Starting A-QMIX Training for {num_episodes} episodes ---")

        self._init_new_gce() # Initialize first GCE

        for episode in range(1, num_episodes + 1):
            ep_start_time = time.time()
            states_dict, _ = self.env.reset() # Initial local states for waiting vehicles
            ep_done = False
            ep_steps = 0
            ep_total_vehicles_started = 0
            ep_qmix_losses_this_episode = []

            while not ep_done:
                if ep_steps % 100 == 0 and ep_steps > 0:
                    print(f"Episode {episode}, Env Step {ep_steps}, Total Steps {self.total_steps}, GCE Buffer: {len(self.gce_buffer)}")

                if self.config.get("log_states", False) and ep_steps > 0 and ep_steps % self.config.get("log_frequency_steps", 100) == 0:
                    # Log global state
                    global_state_to_log, _ = self._get_global_state_and_local_states()
                    self.log_global_state(global_state_to_log, f"Episode {episode}, Step {ep_steps}")

                    # Log a sample local state if there are waiting vehicles
                    if states_dict:
                        # Find first vehicle to log
                        hub_to_log = next(iter(states_dict.keys()), None)
                        if hub_to_log and states_dict[hub_to_log]:
                            veh_to_log = next(iter(states_dict[hub_to_log].keys()), None)
                            if veh_to_log:
                                state_data = states_dict[hub_to_log][veh_to_log]
                                local_state_to_log = state_data['state']
                                self.env.log_local_state(local_state_to_log, hub_to_log, veh_to_log)

                actions_to_env = {}
                for hub_id, vehicles_at_hub_state_data in states_dict.items():
                    if hub_id not in self.agents: continue
                    agent = self.agents[hub_id]
                    actions_to_env[hub_id] = {}
                    for vehicle_id, vehicle_s_a_data in vehicles_at_hub_state_data.items():
                        local_s_for_decision = vehicle_s_a_data['state']
                        action_mask = vehicle_s_a_data['action_mask']
                        
                        action = agent.select_action(local_s_for_decision, action_mask)
                        agent.record_decision(local_s_for_decision, action) # Store (s,a)
                        self.current_gce['initiated_decisions_count'] += 1
                        
                        actions_to_env[hub_id][vehicle_id] = {
                            'action': action, 
                            'state': local_s_for_decision # Env expects state for its internal logic too
                        }
                
                next_states_dict, _, terminated, truncated, info = self.env.step(actions_to_env)
                ep_done = terminated or truncated
                
                # Collect rewards for GCE from completed hops
                if 'completed_experiences' in info:
                    for _, experiences_at_origin in info['completed_experiences'].items():
                        for exp_data in experiences_at_origin:
                            # reward is -hop_time_t_xy
                            self.current_gce['resolved_rewards'].append(exp_data.get('reward', 0.0)) 

                ep_total_vehicles_started += info.get('step_journey_starts', 0)
                
                # Check if GCE should end
                if self._should_end_gce(terminated=ep_done):
                    self._end_current_gce(terminated=ep_done) # This also inits the next GCE

                # Perform A-QMIX update periodically (e.g., every N total_steps)
                if self.total_steps > 0 and self.total_steps % qmix_update_frequency == 0:
                    if len(self.gce_buffer) >= min_gce_buffer_size:
                        qmix_loss = self._update_qmix(qmix_batch_size)
                        if qmix_loss is not None:
                            ep_qmix_losses_this_episode.append(qmix_loss)
                            # Soft update all target networks
                            self._soft_update_mixing_network(polyak)
                            for agent in self.agents.values():
                                agent.soft_update_target_network(polyak)
                
                states_dict = next_states_dict
                ep_steps += 1
                self.total_steps += 1

                if ep_steps >= self.config.get('max_steps_per_episode', 2000):
                    ep_done = True
            
            # End of Episode
            ep_duration = time.time() - ep_start_time
            final_avg_travel_time = info.get('final_average_travel_time', 0)
            final_completed = self.env.total_vehicles_completed # From env
            
            # Calculate avg_travel_time if it wasn't set by the environment (episode ended by max steps)
            if final_avg_travel_time == 0 and final_completed > 0:
                final_avg_travel_time = self.env.total_travel_time / final_completed
                
            completion_rate = final_completed / max(1, ep_total_vehicles_started) if ep_total_vehicles_started > 0 else 0.0

            self.episode_actual_travel_times.append(final_avg_travel_time)
            self.episode_completion_rates.append(completion_rate)
            self.episode_steps.append(ep_steps)
            
            # Decay Epsilon for all agents
            if len(self.gce_buffer) >= min_gce_buffer_size : # Start decay once enough GCEs collected
                 for agent in self.agents.values():
                     agent.decay_epsilon()

            if episode % log_frequency == 0 or episode == num_episodes:
                avg_qmix_loss_ep = np.mean(ep_qmix_losses_this_episode) if ep_qmix_losses_this_episode else float('nan')
                current_eps = self.agents[list(self.agents.keys())[0]].epsilon if self.agents else 0.0
                print(f"\nEpisode {episode}/{num_episodes} | Steps: {ep_steps} | Duration: {ep_duration:.2f}s")
                print(f"  Avg Travel Time: {final_avg_travel_time:.2f}s | Completion Rate: {completion_rate:.3f} ({final_completed}/{ep_total_vehicles_started})")
                print(f"  Avg QMIX Loss (ep): {avg_qmix_loss_ep:.4f} | Epsilon: {current_eps:.4f}")
                print(f"  GCE Buffer Size: {len(self.gce_buffer)}")

            if episode % save_frequency == 0 or episode == num_episodes:
                self.save_models(os.path.join(save_dir, f"episode_{episode}"))
        
        print(f"\n--- A-QMIX Training Finished ---")

        self.save_models(os.path.join(save_dir, "final_aqmix"))
        self.plot_metrics(save_dir=save_dir)

    def save_models(self, directory): # Renamed from save_agents for clarity
        os.makedirs(directory, exist_ok=True)
        for hub_id, agent in self.agents.items():
            agent.save(os.path.join(directory, f"agent_{hub_id}.pt"))
        
        mixing_network_path = os.path.join(directory, "mixing_network.pt")
        torch.save({
            'mixing_network_state_dict': self.mixing_network.state_dict(),
            'target_mixing_network_state_dict': self.target_mixing_network.state_dict(),
            'mixing_optimizer_state_dict': self.mixing_optimizer.state_dict(),
        }, mixing_network_path)
        print(f"Models (agents & mixing net) saved to {directory}")
        
        metrics_path = os.path.join(directory, "training_metrics_aqmix.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump({
                'episode_actual_travel_times': self.episode_actual_travel_times,
                'episode_completion_rates': self.episode_completion_rates,
                'episode_steps': self.episode_steps,
                'mixing_losses': self.mixing_losses, # Save QMIX losses
                'total_steps': self.total_steps,
                'config': self.config
            }, f)
        print(f"Training metrics saved to {metrics_path}")

    def load_models(self, directory): # Renamed from load_agents
        if not os.path.isdir(directory):
             print(f"Error: Load directory not found: {directory}")
             return False
        loaded_count = 0
        for hub_id, agent in self.agents.items():
            filepath = os.path.join(directory, f"agent_{hub_id}.pt")
            if os.path.exists(filepath):
                agent.load(filepath)
                loaded_count += 1
        
        mixing_network_path = os.path.join(directory, "mixing_network.pt")
        if os.path.exists(mixing_network_path):
            checkpoint = torch.load(mixing_network_path, map_location=device)
            self.mixing_network.load_state_dict(checkpoint['mixing_network_state_dict'])
            self.target_mixing_network.load_state_dict(checkpoint['target_mixing_network_state_dict'])
            self.mixing_optimizer.load_state_dict(checkpoint['mixing_optimizer_state_dict'])
            self.mixing_losses = checkpoint.get('mixing_losses', [])
            self.mixing_network.to(device)
            self.target_mixing_network.to(device)
            self.target_mixing_network.eval()
            print(f"Mixing network loaded from {mixing_network_path}")
        else:
            print(f"Warning: Mixing network file not found at {mixing_network_path}")

        if loaded_count > 0: print(f"Loaded {loaded_count} agents from {directory}")
        # Load metrics if needed (similar to previous save_agents)
        #reset epsilon to 0
        for agent in self.agents.values():
            agent.epsilon = 0
        return loaded_count > 0


    def plot_metrics(self, save_dir=None, show=True):
        # ... (Plotting logic as before, can add mixing_losses plot)
        if not self.episode_actual_travel_times:
            print("No training metrics to plot.")
            return

        num_episodes = len(self.episode_actual_travel_times)
        episodes = range(1, num_episodes + 1)
        
        plt.figure(figsize=(18, 16)) # Increased height for 3 rows

        plt.subplot(3, 2, 1) # travel time
        plt.plot(episodes, self.episode_actual_travel_times, label='Avg Travel Time')
        if num_episodes >= 10: plt.plot(episodes[9:], np.convolve(self.episode_actual_travel_times, np.ones(10)/10, mode='valid'), label='10-ep Moving Avg TT', alpha=0.7)
        plt.title('Avg Vehicle Travel Time')
        plt.xlabel('Episode'); plt.ylabel('Time (s)'); plt.grid(True, linestyle=':'); plt.legend()

        plt.subplot(3, 2, 2) # completion rate
        plt.plot(episodes, self.episode_completion_rates, label='Completion Rate')
        if num_episodes >= 10: plt.plot(episodes[9:], np.convolve(self.episode_completion_rates, np.ones(10)/10, mode='valid'), label='10-ep Moving Avg CR', alpha=0.7)
        plt.title('Vehicle Completion Rate'); plt.xlabel('Episode'); plt.ylabel('Rate'); plt.ylim(0, 1.1); plt.grid(True, linestyle=':'); plt.legend()

        plt.subplot(3, 2, 3) # steps per episode
        plt.plot(episodes, self.episode_steps, label='Steps per Episode')
        if num_episodes >= 10: plt.plot(episodes[9:], np.convolve(self.episode_steps, np.ones(10)/10, mode='valid'), label='10-ep Moving Avg Steps', alpha=0.7)
        plt.title('Steps per Episode'); plt.xlabel('Episode'); plt.ylabel('Steps'); plt.grid(True, linestyle=':'); plt.legend()
        
        plt.subplot(3, 2, 4) # Mixing Network Loss
        if self.mixing_losses:
            # Smooth the loss curve
            window_size_mix = max(1, len(self.mixing_losses) // 50 if len(self.mixing_losses) > 50 else 10) 
            smoothed_mixing_losses = np.convolve(self.mixing_losses, np.ones(window_size_mix)/window_size_mix, mode='valid')
            plt.plot(smoothed_mixing_losses, label=f'Smoothed Mixing Loss (Win={window_size_mix})')
        plt.title('QMIX Mixing Network Loss'); plt.xlabel('Training Updates'); plt.ylabel('Loss'); plt.grid(True, linestyle=':'); plt.legend()

        # Plot epsilon if needed (can get from one agent)
        # plt.subplot(3, 2, 5) ...

        plt.suptitle('A-QMIX Training Metrics', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if save_dir:
            save_path = os.path.join(save_dir, "aqmix_training_metrics_plot.png")
            plt.savefig(save_path)
            print(f"Metrics plot saved to {save_path}")
        if show: plt.show()
        plt.close()

    def trace_vehicle_journey(self, max_steps=2000):
        """
        Traces a single vehicle's journey through the network to visualize the learned policy.
        Follows the first vehicle's journey through the network until it reaches its destination.
        
        Args:
            max_steps (int): Maximum simulation steps to run
            
        Returns:
            dict: Summary of the vehicle's journey
        """
        print("\n--- Tracing Vehicle Journey with Learned Policy ---")
        states_dict, _ = self.env.reset()
        
        # Initialize tracking variables
        target_vehicle_id = None
        journey_log = []
        step_count = 0
        terminated = False
        
        while not terminated and step_count < max_steps:
            # Run simulation until we find our first vehicle or process the one we're tracking
            actions_to_env = {}
            
            for hub_id, vehicles_at_hub_state_data in states_dict.items():
                if hub_id not in self.agents:
                    continue
                    
                agent = self.agents[hub_id]
                actions_to_env[hub_id] = {}
                
                for vehicle_id, vehicle_s_a_data in vehicles_at_hub_state_data.items():
                    # Set target_vehicle_id to the first vehicle we see if not already set
                    if target_vehicle_id is None:
                        target_vehicle_id = vehicle_id
                        print(f"Selected vehicle {target_vehicle_id} to track")
                    
                    # If this is our target vehicle, record detailed decision information
                    if vehicle_id == target_vehicle_id:
                        local_s_for_decision = vehicle_s_a_data['state']
                        action_mask = vehicle_s_a_data['action_mask']
                        
                        # Get Q-values for all actions
                        state_tensor = torch.from_numpy(local_s_for_decision).float().unsqueeze(0).to(device)
                        agent.time_network.eval()
                        with torch.no_grad():
                            state_data = {'state': state_tensor}
                            q_values = agent.time_network(state_data)[0].cpu().numpy()
                        
                        # Apply action mask
                        masked_q_values = q_values.copy()
                        if action_mask is not None:
                            masked_q_values[~action_mask] = float('inf')  # Mask invalid actions (higher is worse for time estimation)
                        
                        # Select best action
                        action = np.argmin(masked_q_values)
                        
                        # Get target hub from environment's hub agent, not RL agent
                        env_hub_agent = self.env.hub_agents[hub_id]
                        target_hub = env_hub_agent.get_target_hub_for_action(action)
                        
                        # Record decision details
                        decision_record = {
                            'step': step_count,
                            'vehicle_id': vehicle_id,
                            'current_hub': hub_id,
                            'q_values': q_values.tolist(),
                            'masked_q_values': masked_q_values.tolist(),
                            'selected_action': action,
                            'target_hub': target_hub,
                            'time': self.env.current_time
                        }
                        journey_log.append(decision_record)
                        print(f"\nStep {step_count}: Vehicle {vehicle_id} at hub {hub_id}")
                        print(f"  Q-values: {q_values.round(2)}")
                        print(f"  Selected action {action}: Route to hub {target_hub}")
                    
                    # Process the action for this vehicle
                    local_s_for_decision = vehicle_s_a_data['state']
                    action_mask = vehicle_s_a_data['action_mask']
                    action = agent.select_action(local_s_for_decision, action_mask)
                    
                    actions_to_env[hub_id][vehicle_id] = {
                        'action': action,
                        'state': local_s_for_decision
                    }
            
            # Execute step in environment
            next_states_dict, _, terminated, _, info = self.env.step(actions_to_env)
            
            # Check if target vehicle completed its journey
            if target_vehicle_id is not None and target_vehicle_id in info.get('completed_vehicle_ids', []):
                print(f"\nVehicle {target_vehicle_id} completed its journey!")
                
                # Add completion information
                journey_log.append({
                    'step': step_count,
                    'vehicle_id': target_vehicle_id,
                    'status': 'completed',
                    'total_travel_time': info.get('step_total_travel_time', 0),
                    'time': self.env.current_time
                })
                break
            
            # Update for next step
            states_dict = next_states_dict
            step_count += 1
            
            # If our target vehicle isn't present in the current states but we haven't
            # recorded it as completed, it might be in transition between hubs
            if (target_vehicle_id is not None and 
                all(target_vehicle_id not in vehicles for vehicles in states_dict.values()) and
                target_vehicle_id not in info.get('completed_vehicle_ids', [])):
                #print(f".", end="", flush=True)
                pass
        
        # Summarize journey
        if len(journey_log) > 0:
            decisions_made = sum(1 for entry in journey_log if 'selected_action' in entry)
            print(f"\nJourney summary for vehicle {target_vehicle_id}:")
            print(f"  Total steps: {step_count}")
            print(f"  Decisions made: {decisions_made}")
            if any('status' in entry and entry['status'] == 'completed' for entry in journey_log):
                completion_entry = next(entry for entry in journey_log if 'status' in entry and entry['status'] == 'completed')
                start_time = journey_log[0]['time'] if journey_log else 0
                total_time = completion_entry['time'] - start_time
                print(f"  Total travel time: {total_time:.2f} seconds")
                print(f"  Journey completed: Yes")
            else:
                print(f"  Journey completed: No (max steps reached or simulation terminated)")
            
            # Show hubs visited
            hubs_visited = [entry['current_hub'] for entry in journey_log if 'current_hub' in entry]
            print(f"  Hubs visited: {' -> '.join(hubs_visited)}")
        else:
            print("No journey data collected. Vehicle may not have appeared in the simulation.")
        
        print("--- End of Vehicle Journey Trace ---")
        return journey_log


# --- Example Usage ---
if __name__ == "__main__":
    config = {
        "net_file": "maps/UES_Manhatan.net.xml", # *** UPDATE THIS PATH ***
        "trips_file": "maps/UES_Manhatan_trips_fixed_scaled.xml", # *** UPDATE THIS PATH ***
        "num_hubs": 8,
        "gui": False,
        "scale_factor": 1.0,
        "max_waiting_vehicles": 40,
        "z_order_embedding_dim": 8,
        "num_episodes": 10000,
        "max_steps_per_episode": 3000, # Increased steps
        "lr": 0.00025, # Learning rate for agent nets (Q_a, Attention)
        "gamma": 0.99, # Discount factor for QMIX (no longer 1.0)
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.995, # Slower decay for more exploration
        "polyak": 0.995,
        "min_gce_buffer_size": 200, # Start training QMIX sooner
        "gce_buffer_capacity": 10000,
        "qmix_batch_size": 128, # Larger QMIX batch
        "qmix_update_frequency_steps": 128, # Update QMIX less frequently than individual (if any)
        "save_frequency_episodes": 5,
        "log_frequency_episodes": 1,
        "mixing_hidden_dim": 256, # Increased mixing net capacity
        "mixing_lr": 0.0001,     # Separate LR for mixing net
        "gce_size": 16, # num_hubs * 4
        "gce_max_sim_time": 200,
        "clip_grad_norm": 10.0, # Gradient clipping value
        "timeout_penalty_threshold": 200.0, # For global state heuristic
        "log_states": False, # Log state spaces
        "log_frequency_steps": 100 # Frequency to log states
    }
    config["save_dir"] = "aqmix_toronto" + time.strftime("%Y%m%d_%H%M%S")

    env = None
    try:
        print("Initializing HubRoutingEnv...")
        env = HubRoutingEnv(
            net_file=config["net_file"], trips_file=config["trips_file"],
            num_hubs=config["num_hubs"], gui=config["gui"],
            scale_factor=config["scale_factor"],
            max_waiting_vehicles=config["max_waiting_vehicles"],
            z_order_embedding_dim=config["z_order_embedding_dim"]
        )
        """# Set manual hubs and neighbors (if using custom hub configuration)
        manual_hubs = {"hub_0": '24959524', "hub_1": '20953780', "hub_2": 'cluster_25629241_25629242_29658482_3472412709', "hub_3": '29603394', "hub_4": 'gneJ36', "hub_5": '406072531'}
        env.set_manual_hubs(manual_hubs)
        
        manual_neighbors = {
            "hub_0": ["hub_1", "hub_2"],
            "hub_1": ["hub_0", "hub_3"],
            "hub_2": ["hub_0", "hub_3", "hub_4"],
            "hub_3": ["hub_1", "hub_2", "hub_5"],
            "hub_4": ["hub_2", "hub_5"],
            "hub_5": ["hub_3", "hub_4"]
        }
        env.set_manual_hub_neighbors(manual_neighbors)"""
        
        env.visualize_hubs(save_path="hub_graph_initial.png", hub_radius=600)
        #exit()
        
        print("Initializing AQMIXTrainer...")
        trainer = AQMIXTrainer(env, config)

        # Example: trainer.load_models("path_to_pretrained_models_if_any")

        trainer.train(config["num_episodes"])

    except FileNotFoundError as e:
        print(f"\nError: File not found. Please check paths for net_file or trips_file in config.")
        print(e)
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            print("\nClosing environment...")
            env.close()
        print("Script finished.")
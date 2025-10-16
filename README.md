# Multi-Agent Hub Routing with A-QMIX

Multi-agent reinforcement learning system for hub-based vehicle routing using the Attentive Q-Mixing (A-QMIX) architecture with SUMO traffic simulation.

## Installation

### Prerequisites

1. **SUMO Traffic Simulator**
   - Download from https://sumo.dlr.de/docs/Downloads.php
   - Set `SUMO_HOME` environment variable to your SUMO installation directory

2. **Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Training

```bash
python dqr_multiagent.py
```

### Configuration

Edit the config dictionary in `dqr_multiagent.py`:

```python
config = {
    "net_file": "maps/UES_Manhatan.net.xml",      # Road network
    "trips_file": "maps/UES_Manhatan_trips_fixed_scaled.xml",  # Vehicle trips
    "num_hubs": 8,                     # Number of hub locations
    "num_episodes": 10000,             # Training episodes
    "lr": 0.00025,                     # Learning rate
    "gamma": 0.99,                     # Discount factor
    "gui": False,                      # SUMO GUI (True for visualization)
    # ... see file for all options
}
```

### Available Networks

- `maps/5x6.net.xml` - Small grid (testing)
- `maps/UES_Manhatan.net.xml` - Manhattan network
- `maps/toronto.net.xml` - Toronto network

## Architecture

- **TimeEstimationNetwork**: Individual agent Q-networks
- **AttentionModule**: Aggregates multiple decisions per agent
- **QMixingNetwork**: Combines agent utilities monotonically
- **HubRoutingEnv** (`hub_routing_env.py`): SUMO-based traffic simulation environment

## Training Output

Models and metrics saved to `aqmix_[timestamp]/`:
- `agent_*.pt` - Agent networks
- `mixing_network.pt` - QMIX network
- `training_metrics_aqmix.pkl` - Training history
- `aqmix_training_metrics_plot.png` - Training curves

## State Space

**Local State** (per vehicle decision):
- Z-order embedding of destination hub
- Current hub conditions (speed, processing rate)
- Neighbor hub features (speed, distance, congestion)

**Global State** (system-wide):
- Per-hub: vicinity speed and outgoing congestion
- Network-wide: vehicle count, completion rate, trip inefficiency
- System imbalance: standard deviation of hub vicinity speeds


## References

- [QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485)
- [SUMO - Simulation of Urban MObility](https://sumo.dlr.de/)

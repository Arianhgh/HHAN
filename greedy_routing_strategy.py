import numpy as np
import networkx as nx
import random

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions."""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def greedy_strategy(env, states_dict):
    """
    Implements a greedy routing strategy:
    1. For each vehicle, select the next hub that is closer to its destination hub
    2. Randomly select among available precomputed paths to that hub
    
    Args:
        env: The HubRoutingEnv instance
        states_dict: Dictionary of states for vehicles at each hub
        
    Returns:
        Dictionary of actions in the format {hub_id: {vehicle_id: action_index}}
    """
    sim_actions = {}
    
    # For each hub with waiting vehicles
    for hub_id in states_dict.keys():
        if hub_id not in env.hub_agents:
            continue
        
        # Get the agent at this hub
        agent = env.hub_agents[hub_id]
        if agent.action_space_size == 0:
            continue
            
        # Get vehicles waiting at this hub
        vehicles_at_hub = states_dict.get(hub_id, {}).keys()
        if not vehicles_at_hub:
            continue
            
        # Initialize actions dictionary for this hub
        sim_actions[hub_id] = {}
        
        for vehicle_id in vehicles_at_hub:
            # Skip if vehicle is not in the environment's tracking
            if vehicle_id not in env.vehicles:
                continue
                
            # Get vehicle's destination hub
            dest_hub_id = env.vehicles[vehicle_id]['destination_hub']
            
            # Get destination hub coordinates
            if dest_hub_id not in env.hub_coordinates:
                raise ValueError(f"Destination hub {dest_hub_id} not found in hub_coordinates")
                # If we can't find coordinates, use random action as fallback
                action_mask = env.get_available_actions_for_vehicle(hub_id, vehicle_id)
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    sim_actions[hub_id][vehicle_id] = np.random.choice(valid_actions)
                continue

                
            dest_coords = env.hub_coordinates[dest_hub_id]
            
            # Find neighbor hub closest to the destination
            closest_neighbor_idx = None
            min_distance = float('inf')
            valid_actions_to_closest = []
            
            # Get available actions for this vehicle (accounting for U-turns)
            action_mask = env.get_available_actions_for_vehicle(hub_id, vehicle_id)
            
            # For each possible action from this hub
            for action_idx in range(agent.action_space_size):
                # Skip masked actions
                if not action_mask[action_idx]:
                    continue
                    
                # Get neighbor hub and path details for this action
                next_hub_id, path_idx, _ = agent.get_path_details_for_action(action_idx)
                
                # Skip if neighbor hub coordinates not available
                if next_hub_id not in env.hub_coordinates:
                    continue
                    
                # Calculate distance from neighbor to destination
                neighbor_coords = env.hub_coordinates[next_hub_id]
                distance_to_dest = calculate_distance(neighbor_coords, dest_coords)
                
                # If this is the closest neighbor so far
                if distance_to_dest < min_distance:
                    min_distance = distance_to_dest
                    closest_neighbor_idx = next_hub_id
                    valid_actions_to_closest = [action_idx]
                elif distance_to_dest == min_distance:
                    # If there's a tie, add this action as another option
                    valid_actions_to_closest.append(action_idx)
            
            # If we found valid actions to the closest neighbor
            if valid_actions_to_closest:
                # Randomly select among the valid paths to the closest neighbor
                sim_actions[hub_id][vehicle_id] = random.choice(valid_actions_to_closest)
            else:
                raise ValueError(f"No valid actions to the closest neighbor for vehicle {vehicle_id} at hub {hub_id}")
                # Fallback: choose randomly among all valid actions
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    sim_actions[hub_id][vehicle_id] = np.random.choice(valid_actions)
    
    return sim_actions

def run_greedy_strategy_test(env, max_steps=1000):
    """
    Runs a simple test of the environment using a greedy routing strategy.
    Modified for hub selection approach.
    
    Args:
        env: The HubRoutingEnv instance
        max_steps: Maximum number of simulation steps
        
    Returns:
        Metrics dictionary with results
    """
    print("\n--- Running Greedy Hub Selection Strategy Test ---")
    steps = 0
    done = False
    
    while not done and steps < max_steps:
        # Get states of waiting vehicles at each hub
        waiting_vehicles = {}
        for hub_id in env.active_hubs:
            waiting_vehicles_at_hub = [
                v_id for v_id, v_data in env.vehicles.items()
                if v_data.get('current_hub') == hub_id and 
                not v_data.get('has_next_hub') and 
                v_data.get('waiting_at_hub', False)
            ]
            if waiting_vehicles_at_hub:
                waiting_vehicles[hub_id] = waiting_vehicles_at_hub
        
        # No vehicles waiting, just step simulation
        if not waiting_vehicles:
            _, _, done, truncated, _ = env.step({})
            steps += 1
            continue
        
        # Process actions for waiting vehicles using greedy strategy
        actions = {}
        for hub_id, vehicle_ids in waiting_vehicles.items():
            hub_agent = env.hub_agents[hub_id]
            actions[hub_id] = {}
            
            for vehicle_id in vehicle_ids:
                vehicle = env.vehicles[vehicle_id]
                dest_hub_id = vehicle.get('destination_hub')
                
                # Get state and action mask for this vehicle
                state_result = env._get_dqr_state_for_vehicle(hub_id, vehicle_id)
                if state_result is None: continue
                state, action_mask = state_result
                
                # Default to random action if we can't determine best action
                if not action_mask.any():
                    print(f"No valid actions for vehicle {vehicle_id} at hub {hub_id}")
                    continue
                
                valid_actions = np.where(action_mask)[0]
                best_action_idx = valid_actions[0]  # Default to first valid action
                min_distance = float('inf')
                
                # Find neighbor hub closest to destination in network distance
                for action_idx in valid_actions:
                    # Get target hub for this action
                    next_hub_id = hub_agent.get_target_hub_for_action(action_idx)
                    
                    if next_hub_id is None:
                        continue
                    
                    # Calculate network distance from next hub to destination
                    if next_hub_id in env.hub_network_distances and dest_hub_id in env.hub_network_distances[next_hub_id]:
                        dist = env.hub_network_distances[next_hub_id][dest_hub_id]
                        if dist < min_distance:
                            min_distance = dist
                            best_action_idx = action_idx
                
                actions[hub_id][vehicle_id] = {
                    'action': best_action_idx,
                    'state': state
                }
        
        # Step environment with chosen actions
        _, _, done, truncated, _ = env.step(actions)
        steps += 1
        
        if steps % 10 == 0:
            print(f"Step {steps}, Vehicles: {len(env.vehicles)}, Completed: {env.total_vehicles_completed}")
    
    # Process final metrics
    metrics = {
        'total_steps': steps,
        'total_vehicles_completed': env.total_vehicles_completed,
        'total_vehicles_started': env.total_journey_starts,
        'completion_rate': env.total_vehicles_completed / max(1, env.total_journey_starts),
        'avg_travel_time': env.total_travel_time / max(1, env.total_vehicles_completed)
    }
    
    print("\n--- Greedy Hub Selection Test Results ---")
    print(f"Total vehicles: {env.total_journey_starts}")
    print(f"Completed vehicles: {env.total_vehicles_completed} ({metrics['completion_rate']*100:.1f}%)")
    print(f"Average travel time: {metrics['avg_travel_time']:.2f}s")
    print(f"Total steps: {metrics['total_steps']}")
    
    return metrics 
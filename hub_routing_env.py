import os
import numpy as np
import traci
import sumolib
import networkx as nx
from gymnasium import Env, spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import xml.etree.ElementTree as ET
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from greedy_routing_strategy import run_greedy_strategy_test
import random
from sklearn_extra.cluster import KMedoids

# Add the current directory to the path to find fix_trips.py
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from model.network import RoadNetworkModel
from model.vehicle import Vehicle, Entry

# Import the utility function to fix trips file - use try/except for flexibility
try:
    from fix_trips import fix_trips_file
except ImportError:
    # Define the function here as fallback
    def fix_trips_file(input_file, output_file, scale_factor=1.0, random_depart=False):
        """
        Fix a trips file by adding required ID and depart attributes

        Args:
            input_file: Path to input trips file
            output_file: Path to output trips file
            scale_factor: Scale factor for departure times
            random_depart: Whether to use random departure times
        """
        import random

        try:
            tree = ET.parse(input_file)
            root = tree.getroot()
        except ET.ParseError:
            # Handle case where the file might not be valid XML
            # print(f"Could not parse {input_file} as XML. Creating from scratch.")
            content = open(input_file, 'r').read()
            root = ET.Element("trips")
            if content.startswith("<trips>"):
                content = content[7:]  # Remove opening tag
            if content.endswith("</trips>"):
                content = content[:-8]  # Remove closing tag

            # Manually parse trips
            for i, trip_str in enumerate(content.split("<trip ")):
                if i == 0:  # Skip first empty split
                    continue

                # Create a valid XML fragment
                trip_xml = ET.fromstring(f"<trip {trip_str}")
                root.append(trip_xml)

        # Process each trip element
        for i, trip in enumerate(root.findall('.//trip')):
            # Add ID if missing
            if 'id' not in trip.attrib:
                trip.set('id', str(i))

            # Add depart time if missing
            if 'depart' not in trip.attrib:
                if random_depart:
                    depart_time = random.randint(0, 100) * scale_factor
                else:
                    depart_time = i * scale_factor
                trip.set('depart', str(depart_time))
            elif scale_factor != 1.0:
                # Scale existing depart time
                depart_time = float(trip.attrib['depart']) * scale_factor
                trip.set('depart', str(depart_time))

            # Rename 'origin' to 'from' if needed
            if 'origin' in trip.attrib:
                trip.set('from', trip.attrib['origin'])
                del trip.attrib['origin']

            # Rename 'destination' to 'to' if needed
            if 'destination' in trip.attrib:
                trip.set('to', trip.attrib['destination'])
                del trip.attrib['destination']

        # Write fixed file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        # print(f"Fixed trips file written to {output_file}")
        return output_file

# Remove GATEncoder class - we'll be using a simpler state representation without GAT

#--------------------------------------------------------------------------
# Helper functions for Z-order (Placeholders/Implementations)
#--------------------------------------------------------------------------

def get_z_order_value(x, y, grid_bounds, precision):
    """
    Calculates the Z-order curve value for a point (x, y) using bit interleaving.
    
    Args:
        x, y: Coordinates to encode
        grid_bounds: Dictionary with 'xmin', 'ymin', 'xmax', 'ymax' defining the grid space
        precision: Number of bits to use for each coordinate (max 16 for 32-bit result)
    
    Returns:
        Integer representing the Z-order curve value
    """
    # Normalize coordinates based on grid bounds
    if grid_bounds['xmax'] - grid_bounds['xmin'] == 0 or grid_bounds['ymax'] - grid_bounds['ymin'] == 0:
         return 0  # Avoid division by zero

    # Clamp input coordinates to bounds to ensure valid normalization
    x = max(grid_bounds['xmin'], min(x, grid_bounds['xmax']))
    y = max(grid_bounds['ymin'], min(y, grid_bounds['ymax']))

    norm_x = (x - grid_bounds['xmin']) / (grid_bounds['xmax'] - grid_bounds['xmin'])
    norm_y = (y - grid_bounds['ymin']) / (grid_bounds['ymax'] - grid_bounds['ymin'])

    # Scale to integer based on precision
    precision = min(16, precision)  # Limit to 16 bits (for 32-bit result)
    scale = (1 << precision) - 1
    int_x = int(norm_x * scale)
    int_y = int(norm_y * scale)

    # Interleave bits (Morton code)
    z = 0
    for i in range(precision):
        z |= ((int_x & (1 << i)) << i) | ((int_y & (1 << i)) << (i + 1))

    return z

# Remove get_k_shortest_paths_yen function - we'll use SUMO's dynamic routing instead

#--------------------------------------------------------------------------
# Modified HubAgent Class
#--------------------------------------------------------------------------
class HubAgent:
    """
    A hub agent responsible for routing decisions at a specific hub.
    Modified for Hub Selection DQR.
    """
    def __init__(self, hub_id, hub_node):
        self.hub_id = hub_id
        self.hub_node = hub_node # SUMO Junction ID

        # Store only neighbors (successor hubs)
        self.neighbors = [] # List of neighboring hub IDs (sorted)

        # Action space size depends on the number of neighbors
        self.action_space_size = 0

        # Track metrics for this hub
        self.num_vehicles_routed = 0
        self.total_routing_time = 0

    def set_neighbors(self, neighbors):
        """Set neighbors for this hub agent."""
        self.neighbors = sorted(neighbors) # Sort for consistent indexing
        self.action_space_size = len(self.neighbors)

    def get_action_space(self):
        """Returns the size of the discrete action space."""
        return spaces.Discrete(self.action_space_size)

    def get_target_hub_for_action(self, action_index):
        """Gets the target hub ID for a given action index."""
        if 0 <= action_index < self.action_space_size:
            return self.neighbors[action_index]
        else:
            return None # Invalid action

#--------------------------------------------------------------------------
# Modified HubRoutingEnv Class
#--------------------------------------------------------------------------
class HubRoutingEnv(Env):
    """
    Environment for hub-based vehicle routing using SUMO.
    Modified for Multi-Path Hub Graph and DQR State/Action.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # In file: hubroutingenvnew.py

    def __init__(self,
                net_file,
                trips_file,
                hub_selection_method="betweenness",
                num_hubs=4,
                simulation_step=1.0,
                gui=False,
                max_waiting_vehicles=10,
                scale_factor=4,
                random_depart=False,
                # Z-Order Embedding Hyperparameters
                z_order_precision=10, # Precision for Z-order calculation
                z_order_embedding_dim=8,
                max_vehicles=1300):    # Maximum number of vehicles allowed in the system
        """
        Initialize the hub routing environment.

        Args:
            net_file: Path to SUMO network file.
            trips_file: Path to SUMO trips file.
            hub_selection_method: Method to select hubs ('betweenness', 'degree', 'combined').
            num_hubs: Number of hubs to select in the network.
            simulation_step: SUMO simulation step size.
            gui: Whether to show SUMO GUI during simulation.
            max_waiting_vehicles: Maximum waiting vehicles at a hub for normalization.
            scale_factor: Scale factor for trip departure times.
            random_depart: Whether to use random departure times.
            z_order_precision: Bit precision for Z-order curve calculation.
            z_order_embedding_dim: Dimension for the Z-order location embedding.
            max_vehicles: Maximum number of vehicles allowed in the system at once.
        """
        super().__init__()

        self.net_file = net_file
        self.trips_file = trips_file
        self.simulation_step = simulation_step
        self.gui = gui
        self.scale_factor = scale_factor
        self.max_waiting_vehicles = max_waiting_vehicles
        self.random_depart = random_depart
        self.max_vehicles = max_vehicles  # Store max vehicles limit

        # Hyperparameters
        self.z_order_precision = z_order_precision
        self.z_order_embedding_dim = z_order_embedding_dim
        
        # Tracking for pending experiences (for delayed rewards)
        self.pending_experiences = {} # {hub_id: {vehicle_id: {'state': state, 'action': action, 'next_state': next_state, 'reward': None}}}
        
        # Keep track of temporary files to clean up
        self.temp_files = []

        # Initialize SUMO
        self._init_sumo()

        # Load network model and SUMO graph
        self.network_model = RoadNetworkModel(os.path.dirname(net_file), os.path.basename(net_file))
        self.sumo_graph = self.network_model.graph # Underlying SUMO graph
        
        # Precompute all-pairs shortest path distances
        self.all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path_length(self.sumo_graph))
        
        # Add network parsing with SUMOLIB - important for coordinate extraction
        self.node_positions = {}  # Initialize empty node positions
        try:
            import sumolib
            self.sumo_net = sumolib.net.readNet(net_file)
            
            # Add position data to sumo_graph and node_positions from sumolib
            nodes_with_pos = 0
            
            for node in self.sumo_net.getNodes():
                node_id = node.getID()
                x, y = node.getCoord()
                
                # Add to node_positions
                self.node_positions[node_id] = (x, y)
                
                # Also add to sumo_graph if the node exists there
                if node_id in self.sumo_graph.nodes:
                    self.sumo_graph.nodes[node_id]['pos'] = (x, y)
                    self.sumo_graph.nodes[node_id]['x'] = x
                    self.sumo_graph.nodes[node_id]['y'] = y
                    nodes_with_pos += 1
            
        except Exception as e:
            pass

        # Select hubs
        self.hubs = self._select_hubs(hub_selection_method, num_hubs)
        
        # Now get hub coordinates from the selected hubs 
        self.hub_coordinates = {}
        for hub_id, node_id in self.hubs.items():
            if node_id in self.node_positions:
                self.hub_coordinates[hub_id] = self.node_positions[node_id]

        # Create basic hub agents (neighbors set later)
        self.hub_agents = {hub_id: HubAgent(hub_id, node_id)
                        for hub_id, node_id in self.hubs.items()}

        # Create the Hub Graph (now simplified to just hub connectivity)
        self.hub_graph = self._create_hub_graph()

        # Get ingoing edges for each hub
        self._get_hub_ingoing_edges()

        # Populate agents with neighbors from the hub_graph
        self._populate_agent_neighbors()
        
        # Calculate max_neighbors for state padding
        if self.hub_graph.nodes():
            self.max_neighbors = max(len(list(self.hub_graph.successors(hub_id))) 
                            for hub_id in self.hub_graph.nodes())
        else:
            self.max_neighbors = 0
        print(f"Maximum number of neighbors for any hub: {self.max_neighbors}")
        
        # Precompute hub-to-hub shortest path distances for state representation
        self.hub_network_distances = {}
        for source_hub_id, source_node in self.hubs.items():
            self.hub_network_distances[source_hub_id] = {}
            for target_hub_id, target_node in self.hubs.items():
                if source_hub_id != target_hub_id:
                    # Get shortest path distance from network
                    if source_node in self.all_pairs_shortest_path and target_node in self.all_pairs_shortest_path[source_node]:
                        distance = self.all_pairs_shortest_path[source_node][target_node]
                    else:
                        # Default large distance if no path exists
                        distance = 1000.0  
                    self.hub_network_distances[source_hub_id][target_hub_id] = distance

        # Initialize Z-order grid bounds and embedding layer
        self._init_z_order_embedding()

        # Initialize vehicles dict
        self.vehicles = {}

        # Track active hubs
        self.active_hubs = set()

        # Track metrics
        self.total_travel_time = 0
        self.total_vehicles_completed = 0
        self.total_journey_starts = 0
        self.waiting_vehicles_per_hub = {hub_id: 0 for hub_id in self.hub_agents}
        self.routed_to_final_vehicles = set()
        self.completed_vehicle_ids = set() # MODIFICATION: Added to track completed vehicles across an episode

        # Add tracking for trip inefficiency KPI
        self.completed_trip_inefficiencies = []

        # Track simulation time
        self.current_time = 0

        # --- Helper mappings for indexing ---
        self.hub_id_list = list(self.hubs.keys())
        self.hub_id_to_idx = {hub_id: i for i, hub_id in enumerate(self.hub_id_list)}
        self.hub_idx_to_id = {i: hub_id for i, hub_id in enumerate(self.hub_id_list)}

        # Define action and observation spaces (placeholders, specific to agent/vehicle)
        # The DQR agent will need to know the state size and action size dynamically.
        self.observation_space = None # Determined by _get_dqr_state_for_vehicle output size
        self.action_space = None      # Each agent uses hub_agent.get_action_space()

    def _init_sumo(self):
        """Initialize SUMO and load the network."""
        try:
            if self.gui:
                sumo_binary = sumolib.checkBinary('sumo-gui')
            else:
                sumo_binary = sumolib.checkBinary('sumo')
        except Exception as e:
            raise RuntimeError(f"Error finding SUMO binary. Please ensure SUMO is installed and in your PATH.\nError: {str(e)}")

        # Check if the net file exists
        if not os.path.exists(self.net_file):
            raise FileNotFoundError(f"Network file not found: {self.net_file}")
        
        # Check if the trips file exists
        if not os.path.exists(self.trips_file):
            raise FileNotFoundError(f"Trips file not found: {self.trips_file}")

        # Fix the trips file to ensure it has required attributes
        try:
            fixed_trips_file = os.path.join(
                os.path.dirname(self.trips_file),
                f"{os.path.basename(self.trips_file).split('.')[0]}_fixed.xml"
            )
            fixed_trips_file = fix_trips_file(
                self.trips_file,
                fixed_trips_file,
                scale_factor=1.0,  # Use 1.0 for fixing, we'll scale separately
                random_depart=self.random_depart
            )
            self.temp_files.append(fixed_trips_file)

            # Now apply scaling if needed
            if self.scale_factor != 1.0:
                scaled_trips_file = os.path.join(
                    os.path.dirname(fixed_trips_file),
                    f"{os.path.basename(fixed_trips_file).split('.')[0]}_scaled.xml"
                )

                # Directly scale the fixed file using our scaling function
                tree = ET.parse(fixed_trips_file)
                root = tree.getroot()

                # Scale depart times
                for trip in root.findall('.//trip'):
                    if 'depart' in trip.attrib:
                        depart_time = float(trip.attrib['depart']) * self.scale_factor
                        trip.attrib['depart'] = str(depart_time)

                # Write scaled file
                tree.write(scaled_trips_file, encoding='utf-8', xml_declaration=True)
                self.temp_files.append(scaled_trips_file)
                trips_to_use = scaled_trips_file
            else:
                trips_to_use = fixed_trips_file

            # print(f"Using trips file: {trips_to_use}")

            sumo_cmd = [
                sumo_binary,
                '-n', self.net_file,
                '--route-files', trips_to_use,
                '--step-length', str(self.simulation_step),
                '--no-warnings', 'true',
                '--no-step-log', 'true',
                '--max-depart-delay', '1000',
                '--time-to-teleport', '-1',
                '--max-num-vehicles', str(self.max_vehicles)
            ]

            try:
                traci.start(sumo_cmd)
                # Command-line parameter --max-num-vehicles will handle the vehicle limit
            except traci.exceptions.FatalTraCIError as e:
                raise RuntimeError(f"Error starting SUMO simulation: {str(e)}")
            
        except Exception as e:
            # Clean up any temporary files created
            for temp_file in self.temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
            raise RuntimeError(f"Error during SUMO initialization: {str(e)}")

        # Add debug check for graph integrity after initialization
        self._check_graph_integrity()
        
        return
        
    def _check_graph_integrity(self):
        """Debug function to check graph and node data structures."""
        # print("\n--- DEBUG: GRAPH INTEGRITY CHECK ---")
        
        if not hasattr(self, 'sumo_graph') or self.sumo_graph is None:
            # print("ERROR: sumo_graph is not initialized")
            return
            
        # print(f"sumo_graph type: {type(self.sumo_graph)}")
        # print(f"Number of nodes: {len(self.sumo_graph.nodes())}")
        # print(f"Number of edges: {len(self.sumo_graph.edges())}")
        
        # Check for node attributes
        if self.sumo_graph.nodes():
            sample_node = next(iter(self.sumo_graph.nodes()))
            # print(f"Sample node: {sample_node}")
            # print(f"Sample node attributes: {self.sumo_graph.nodes[sample_node]}")
            # print(f"Available node attributes: {list(self.sumo_graph.nodes[sample_node].keys())}")
            
            # Check for position attributes
            pos_attrs = ['pos', 'x', 'y']
            for attr in pos_attrs:
                if attr in self.sumo_graph.nodes[sample_node]:
                    # print(f"Position attribute '{attr}' found with value: {self.sumo_graph.nodes[sample_node][attr]}")
                    pass
        
        # Check network_model
        if hasattr(self, 'network_model') and self.network_model is not None:
            # print(f"network_model edges count: {len(self.network_model.edges)}")
            if self.network_model.edges:
                sample_edge = next(iter(self.network_model.edges.values()))
                # print(f"Sample edge: {sample_edge}")
                # print(f"Sample edge shape: {sample_edge.shape}")
                if hasattr(sample_edge.shape, 'get_center'):
                    # print(f"Sample edge center: {sample_edge.shape.get_center()}")
                    pass
                else:
                    # print("ERROR: edge.shape.get_center() method not available")
                    pass
        else:
            # print("ERROR: network_model is not initialized")
            pass
            
        # Check node_positions calculation logic
        test_node_positions = {}
        count_with_pos = 0
        count_without_pos = 0
        
        # Check first 10 nodes for position data
        for i, node_id in enumerate(list(self.sumo_graph.nodes())[:10]):
            node_pos = None
            if 'pos' in self.sumo_graph.nodes[node_id]:
                node_pos = self.sumo_graph.nodes[node_id]['pos']
                method = "pos attribute"
            elif 'x' in self.sumo_graph.nodes[node_id] and 'y' in self.sumo_graph.nodes[node_id]:
                node_pos = (self.sumo_graph.nodes[node_id]['x'], self.sumo_graph.nodes[node_id]['y'])
                method = "x/y attributes"
            
            if node_pos:
                test_node_positions[node_id] = node_pos
                count_with_pos += 1
                # print(f"Node {node_id} position {node_pos} found via {method}")
            else:
                count_without_pos += 1
                
                # Try alternative from network_model
                connected_edges = []
                for edge in self.network_model.edges.values():
                    if edge.from_id == node_id or edge.to_id == node_id:
                        connected_edges.append(edge)
                
                if connected_edges:
                    alt_pos = connected_edges[0].shape.get_center()
                    # print(f"Node {node_id} alternative position {alt_pos} from edge shape")
                else:
                    # print(f"Node {node_id} has no position data and no connected edges")
                    pass
        
        # print(f"Test scan: {count_with_pos} nodes with position, {count_without_pos} without")

    def _get_hub_coordinates(self):
        """Get XY coordinates for all selected hubs from sumolib."""
        self.hub_coordinates = {}
        self.node_positions = {} # Store positions for all nodes
        
        # print("\n--- DEBUG: NODE POSITIONS ---")
        # print(f"Total nodes in sumo_graph: {len(self.sumo_graph.nodes())}")
        
        # Get node positions from sumolib if available
        if hasattr(self, 'sumo_net') and self.sumo_net:
            # print("Using sumolib for accurate node positions")
            # Get all node positions from sumolib
            for node in self.sumo_net.getNodes():
                node_id = node.getID()
                x, y = node.getCoord()
                self.node_positions[node_id] = (x, y)
                
            # print(f"Loaded {len(self.node_positions)} node positions from sumolib")
        else:
            # print("WARNING: sumolib not available, falling back to graph attributes")
            # Fallback to graph attributes if sumolib not available
            for node_id in self.sumo_graph.nodes():
                x, y = None, None
                # Try both data access patterns to be safe
                if 'pos' in self.sumo_graph.nodes[node_id]:
                    x, y = self.sumo_graph.nodes[node_id]['pos']
                elif 'x' in self.sumo_graph.nodes[node_id] and 'y' in self.sumo_graph.nodes[node_id]:
                    x = self.sumo_graph.nodes[node_id]['x']
                    y = self.sumo_graph.nodes[node_id]['y']
                    
                if x is not None and y is not None:
                    self.node_positions[node_id] = (x, y)
        
        # print(f"Found positions for {len(self.node_positions)}/{len(self.sumo_graph.nodes())} nodes")
        
        # Get hub coordinates from node positions
        missing_hub_positions = []
        for hub_id, node_id in self.hubs.items():
            if node_id in self.node_positions:
                self.hub_coordinates[hub_id] = self.node_positions[node_id]
            else:
                missing_hub_positions.append((hub_id, node_id))
        
        # print(f"Extracted coordinates for {len(self.hub_coordinates)}/{len(self.hubs)} hubs")
        
        if missing_hub_positions:
            # print("Missing hub positions:")
            for hub_id, node_id in missing_hub_positions:
                # print(f"  Hub {hub_id} (node {node_id}): No position found")
                # Try to get position from network_model edges as a fallback
                connected_edges = []
                for edge in self.network_model.edges.values():
                    if edge.from_id == node_id or edge.to_id == node_id:
                        connected_edges.append(edge)
                
                if connected_edges:
                    pos = connected_edges[0].shape.get_center()
                    self.hub_coordinates[hub_id] = pos
                    self.node_positions[node_id] = pos
                    # print(f"  Found fallback position from edge: {pos}")
        
        # Debug: print hub coordinates
        # print("Hub coordinates:")
        # for hub_id, coords in self.hub_coordinates.items():
        #    print(f"  Hub {hub_id} (node {self.hubs[hub_id]}): {coords}")
            
        return self.hub_coordinates
        
    def _get_hub_ingoing_edges(self):
        """Find all ingoing edges for each hub node."""
        self.hub_ingoing_edges = {}
        
        for hub_id, node_id in self.hubs.items():
            ingoing_edges = []
            
            # Get all predecessors (nodes with edges leading to this hub node)
            for pred in self.sumo_graph.predecessors(node_id):
                edge_data = self.sumo_graph.get_edge_data(pred, node_id)
                
                # Extract edge ID based on graph structure
                if isinstance(edge_data, dict):
                    if 0 in edge_data:
                        edge_data_item = edge_data[0]
                        if 'edge' in edge_data_item and hasattr(edge_data_item['edge'], 'id'):
                            edge_id = edge_data_item['edge'].id
                            ingoing_edges.append(edge_id)
                        elif 'id' in edge_data_item:
                            edge_id = edge_data_item['id']
                            ingoing_edges.append(edge_id)
                    elif 'edge' in edge_data:
                        if hasattr(edge_data['edge'], 'id'):
                            edge_id = edge_data['edge'].id
                            ingoing_edges.append(edge_id)
                    elif 'id' in edge_data:
                        edge_id = edge_data['id']
                        ingoing_edges.append(edge_id)
                elif hasattr(edge_data, 'id'):
                    edge_id = edge_data.id
                    ingoing_edges.append(edge_id)
            self.hub_ingoing_edges[hub_id] = ingoing_edges
            # print(f"Hub {hub_id} (node {node_id}) has {len(ingoing_edges)} ingoing edges")
        
        return self.hub_ingoing_edges


    def _select_hubs(self, method="k-medoids", num_hubs=4, min_distance=None):
        """
        Selects intersections to serve as hubs using the K-Medoids (PAM) algorithm,
        first filtering for significant nodes.

        This method first filters for nodes with a degree of 5 or more. It then
        clusters these significant nodes based on shortest path distances and
        selects the most central node (medoid) from each cluster.

        Args:
            method (str): The selection method. Set to 'k-medoids' by default.
            num_hubs (int): The number of hubs (clusters) to select.
            min_distance (int, optional): This argument is not used by the K-Medoids
                                         implementation.

        Returns:
            Dictionary of hub_id -> node_id
        """
        print(f"\n--- Selecting {num_hubs} hubs using K-Medoids (PAM) with node degree filter ---")

        if KMedoids is None:
            raise ImportError(
                "scikit-learn-extra is not installed. Please install it using "
                "'pip install scikit-learn-extra' to use the K-Medoids method."
            )

        all_nodes = list(self.sumo_graph.nodes())
        if not all_nodes:
            print("Error: No nodes in the graph to select hubs from.")
            return {}

        # --- NEW: Filter nodes by degree ---
        print("Filtering for significant nodes (degree >= 5)...")
        nodes = sorted([
            node for node in all_nodes if self.sumo_graph.degree(node) >= 5
        ])

        # Fallback if not enough nodes meet the criteria
        if len(nodes) < num_hubs:
            print(f"Warning: Only {len(nodes)} nodes have a degree of 5 or more, but {num_hubs} hubs are requested.")
            print("Falling back to using all nodes for selection.")
            nodes = sorted(all_nodes)

        print(f"Found {len(nodes)} candidate nodes for hub selection.")
        # --- End of new filtering logic ---

        # 1. Create a mapping from node ID to a matrix index
        node_to_idx = {node_id: i for i, node_id in enumerate(nodes)}

        # 2. Create the full distance matrix for the (potentially filtered) nodes
        num_nodes = len(nodes)
        distance_matrix = np.full((num_nodes, num_nodes), fill_value=999999.0)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    distance_matrix[i, j] = 0
                    continue

                source_node = nodes[i]
                target_node = nodes[j]

                dist = self.all_pairs_shortest_path.get(source_node, {}).get(target_node)
                if dist is not None:
                    distance_matrix[i, j] = dist

        # 3. Instantiate and run the K-Medoids algorithm
        print("Clustering nodes to find medoids... this may take a moment.")
        kmedoids = KMedoids(
            n_clusters=num_hubs,
            metric='precomputed',
            method='pam',
            init='k-medoids++',
            random_state=42
        )
        kmedoids.fit(distance_matrix)

        # 4. Extract the results
        medoid_indices = kmedoids.medoid_indices_
        selected_nodes = [nodes[i] for i in medoid_indices]

        # Create the hub dictionary in the required format
        hubs = {f"hub_{i}": node for i, node in enumerate(sorted(selected_nodes))}
        print(f"Selected {len(hubs)} hubs (medoids): {list(hubs.values())}")

        return hubs


    def _get_edge_by_id(self, edge_id):
        """Helper method to get an edge by ID from the network model."""
        return self.network_model.edges.get(edge_id)

    def _create_hub_graph(self, k_neighbors=3):
        """
        Creates the higher-level graph connecting hub nodes by finding the
        k-nearest neighbors for each hub based on shortest path travel time.

        This method replaces the simple distance threshold with a more robust
        network-aware approach, ensuring neighbors are topologically close, not
        just geographically close.

        Args:
            k_neighbors (int): The number of nearest neighbors to connect to from each hub.
        """
        hub_graph = nx.DiGraph()

        # Add hubs as nodes
        for hub_id in self.hubs.keys():
            hub_graph.add_node(hub_id, waiting_vehicles=0.0, local_congestion=0.0)

        print(f"\nCreating hub graph by finding the {k_neighbors}-nearest neighbors for each hub...")
        edge_count = 0

        # For each hub, find its k-nearest neighbors based on the pre-computed shortest paths
        for hub_id_i, node_i in self.hubs.items():
            
            # Calculate network travel distance to all other hubs
            distances_to_others = []
            for hub_id_j, node_j in self.hubs.items():
                if hub_id_i == hub_id_j:
                    continue
                
                # Get the pre-computed shortest path distance (travel time) from the network graph
                dist = self.all_pairs_shortest_path.get(node_i, {}).get(node_j)
                
                if dist is not None:
                    distances_to_others.append((hub_id_j, dist))

            # Sort other hubs by the travel distance and select the top 'k'
            sorted_neighbors = sorted(distances_to_others, key=lambda x: x[1])
            k_nearest = sorted_neighbors[:k_neighbors]
            
            # --- ADDED: Logging header for the current hub ---
            print(f"Connections for {hub_id_i}:")

            # Add a directed edge from the current hub to each of its k-nearest neighbors
            for neighbor_hub_id, travel_time in k_nearest:
                
                # Calculate physical distance between hub coordinates
                physical_distance = 0.0
                if hub_id_i in self.hub_coordinates and neighbor_hub_id in self.hub_coordinates:
                    coord_i = self.hub_coordinates[hub_id_i]
                    coord_j = self.hub_coordinates[neighbor_hub_id]
                    # Euclidean distance
                    physical_distance = ((coord_i[0] - coord_j[0])**2 + (coord_i[1] - coord_j[1])**2)**0.5
                
                # --- MODIFIED: Log both network distance and physical distance ---
                print(f"  - Connecting to {neighbor_hub_id} (Network Distance: {travel_time:.2f}, Physical Distance: {physical_distance:.2f}m)")

                hub_graph.add_edge(
                    hub_id_i, neighbor_hub_id,
                    static_length=travel_time,  # The 'weight' from Dijkstra is the most relevant metric
                    static_travel_time=travel_time,
                    current_travel_time=travel_time, # Initialize with static estimate
                    current_vehicle_count=0
                )
                edge_count += 1

        print(f"\nCreated hub graph with {hub_graph.number_of_nodes()} nodes and {edge_count} edges.")
        return hub_graph

    def _populate_agent_neighbors(self):
         """Populates HubAgent instances with neighbor lists."""
         for hub_id_i in self.hub_agents:
             agent = self.hub_agents[hub_id_i]
             neighbors = list(self.hub_graph.successors(hub_id_i))
             unique_neighbors = sorted(list(set(neighbors))) # Get unique neighbor hubs

             agent.set_neighbors(unique_neighbors)
             print(f"Agent {hub_id_i}: Neighbors={agent.neighbors}, Action Space Size={agent.action_space_size}")

    def _init_z_order_embedding(self):
        """Initializes bounds for Z-order and the embedding layer."""
        if not self.hub_coordinates:
            # print("Warning: Hub coordinates not available for Z-order initialization.")
            self.z_order_bounds = None
            self.z_order_embedding = None
            return

        # Calculate bounds from hub coordinates
        min_x = min(p[0] for p in self.hub_coordinates.values())
        max_x = max(p[0] for p in self.hub_coordinates.values())
        min_y = min(p[1] for p in self.hub_coordinates.values())
        max_y = max(p[1] for p in self.hub_coordinates.values())
        padding = 10 # Add small padding

        self.z_order_bounds = {
            'xmin': min_x - padding, 'ymin': min_y - padding,
            'xmax': max_x + padding, 'ymax': max_y + padding
        }

        # Calculate max possible Z-order value (vocab size for embedding)
        # This depends heavily on the Z-order implementation and precision
        # Placeholder: Assume a reasonable upper bound or calculate based on precision
        max_z_value_approx = 1 << (2 * self.z_order_precision) # Rough estimate
        self.z_order_vocab_size = max_z_value_approx

        # Create embedding layer
        self.z_order_embedding = nn.Embedding(
            num_embeddings=self.z_order_vocab_size,
            embedding_dim=self.z_order_embedding_dim
        )
        # print(f"Initialized Z-order embedding: bounds={self.z_order_bounds}, vocab_size~={self.z_order_vocab_size}")


    def _get_z_order_embedding(self, hub_id):
        """Computes the Z-order value and returns its embedding."""
        if not self.z_order_bounds or not self.z_order_embedding:
            return torch.zeros(self.z_order_embedding_dim, dtype=torch.float)

        if hub_id not in self.hub_coordinates:
             # print(f"Warning: Coordinates for hub {hub_id} not found for Z-order.")
             return torch.zeros(self.z_order_embedding_dim, dtype=torch.float)

        x, y = self.hub_coordinates[hub_id]
        z_value = get_z_order_value(x, y, self.z_order_bounds, self.z_order_precision)

        # Ensure z_value is within embedding range (clamp if necessary)
        z_value_clamped = max(0, min(z_value, self.z_order_vocab_size - 1))
        z_tensor = torch.tensor([z_value_clamped], dtype=torch.long)

        return self.z_order_embedding(z_tensor).squeeze(0)


    def _update_hub_graph_dynamic_features(self):
        """
        Updates the dynamic features of the simplified hub graph.
        This method is simplified from the previous path-based implementation.
        """
        if not hasattr(self, 'hub_graph') or not self.hub_graph:
            return

        # 1. Update Node Features (waiting vehicles, local congestion)
        for hub_id, node_data in self.hub_graph.nodes(data=True):
            # Waiting vehicles (normalized)
            node_data['waiting_vehicles'] = min(self.waiting_vehicles_per_hub.get(hub_id, 0),
                                              self.max_waiting_vehicles) / max(1, self.max_waiting_vehicles)

            # Local congestion around the hub
            hub_node = self.hubs[hub_id]
            incoming_edges = []
            outgoing_edges = []
            
            # Get incoming SUMO edges
            for u, v in self.sumo_graph.in_edges(hub_node):
                if not self.sumo_graph.has_edge(u, v): continue
                edge_data = self.sumo_graph.get_edge_data(u, v)
                edge_id = None
                if isinstance(edge_data, dict):
                    if 0 in edge_data and 'edge' in edge_data[0]:
                        if hasattr(edge_data[0]['edge'], 'id'):
                            edge_id = edge_data[0]['edge'].id
                if edge_id:
                    incoming_edges.append(edge_id)
            
            # Get outgoing SUMO edges
            for u, v in self.sumo_graph.out_edges(hub_node):
                if not self.sumo_graph.has_edge(u, v): continue
                edge_data = self.sumo_graph.get_edge_data(u, v)
                edge_id = None
                if isinstance(edge_data, dict):
                    if 0 in edge_data and 'edge' in edge_data[0]:
                        if hasattr(edge_data[0]['edge'], 'id'):
                            edge_id = edge_data[0]['edge'].id
                if edge_id:
                    outgoing_edges.append(edge_id)
            
            # Calculate local congestion based on travel times
            total_norm_tt = 0
            valid_edges = 0
            
            for edge_id in incoming_edges + outgoing_edges:
                try:
                    edge_obj = self._get_edge_by_id(edge_id)
                    if not edge_obj: continue
                    
                    current_tt = traci.edge.getTraveltime(edge_id)
                    free_flow_tt = edge_obj.length / max(0.1, edge_obj.flow_speed)
                    norm_tt = current_tt / max(1.0, free_flow_tt)
                    
                    total_norm_tt += norm_tt
                    valid_edges += 1
                except:
                    continue
            
            if valid_edges > 0:
                node_data['local_congestion'] = min(1.0, total_norm_tt / valid_edges / 10.0)
            else:
                node_data['local_congestion'] = 0.0

        # 2. Update Edge Features (direct hub-to-hub connections)
        for u, v, edge_data in self.hub_graph.edges(data=True):
            # Get travel time estimate between hubs using SUMO findRoute
            from_node = self.hubs[u]
            to_node = self.hubs[v]
            
            # Find ingoing edges to the target hub
            to_ingoing_edges = self.hub_ingoing_edges.get(v, [])
            
            # Find outgoing edges from source hub
            from_outgoing_edges = []
            for succ in self.sumo_graph.successors(from_node):
                edge_data = self.sumo_graph.get_edge_data(from_node, succ)
                edge_id = None
                if isinstance(edge_data, dict):
                    if 0 in edge_data and 'edge' in edge_data[0]:
                        if hasattr(edge_data[0]['edge'], 'id'):
                            edge_id = edge_data[0]['edge'].id
                if edge_id:
                    from_outgoing_edges.append(edge_id)
            
            # Calculate current travel time using SUMO findRoute if we have edges
            if from_outgoing_edges and to_ingoing_edges:
                min_travel_time = float('inf')
                
                for from_edge in from_outgoing_edges:
                    for to_edge in to_ingoing_edges:
                        try:
                            # Try to find route between edges
                            route = traci.simulation.findRoute(from_edge, to_edge)
                            if route and route.edges and len(route.edges) > 0:
                                if route.travelTime < min_travel_time:
                                    min_travel_time = route.travelTime
                        except:
                            continue
                
                if min_travel_time < float('inf'):
                    # Update edge with current travel time
                    edge_data['current_travel_time'] = min_travel_time
                else:
                    # Fallback to static travel time
                    edge_data['current_travel_time'] = edge_data.get('static_travel_time', 100.0)
            
            # Normalize travel time
            static_tt = edge_data.get('static_travel_time', 100.0)
            current_tt = edge_data.get('current_travel_time', static_tt)
            edge_data['norm_travel_time'] = min(10.0, current_tt / max(1.0, static_tt)) / 10.0




    def _get_dqr_state_for_vehicle(self, hub_id, vehicle_id):
        """
        Constructs the new flow-based DQR state vector for a specific vehicle at a hub.
        
        Local State Vector Components:
        - Goal Representation: destination_z_order_embedding
        - Current Hub Conditions: current_hub_vicinity_speed_normalized, current_hub_outgoing_congestion_ratio
        - Padded Neighbor Features (for each potential neighbor N_k):
          - travel_time_to_Nk: Real-time travel time to neighbor
          - Nk_incoming_congestion_ratio: Congestion on edges entering neighbor hub
          - distance_from_Nk_to_destination: Network distance from neighbor to final destination

        Args:
            hub_id: The ID of the hub where the agent is located.
            vehicle_id: The ID of the vehicle needing routing.

        Returns:
            tuple (state_vector, action_mask) or None.
        """
        if vehicle_id not in self.vehicles or hub_id not in self.hub_agents:
            return None
        
        vehicle = self.vehicles[vehicle_id]
        dest_hub_id = vehicle['destination_hub']
        agent = self.hub_agents[hub_id]
        action_mask = self.get_available_actions_for_vehicle(hub_id, vehicle_id)

        # 1. Goal Representation: Destination Z-Order Embedding (Unchanged)
        dest_loc_embedding = self._get_z_order_embedding(dest_hub_id).detach().numpy()

        # 2. Current Hub Conditions (Flow-based Local Awareness)
        # A. Current Hub Vicinity Speed Normalized
        current_hub_vicinity_speed = self.get_hub_vicinity_speed(hub_id)
        
        # B. Current Hub Outgoing Congestion Ratio 
        current_hub_outgoing_congestion = self.get_hub_outgoing_congestion_ratio(hub_id)

        # 3. Padded Neighbor Features (Flow-based Consequence Awareness)
        sorted_neighbors = agent.neighbors
        padded_neighbor_features = []
        
        for i in range(self.max_neighbors):
            if i < len(sorted_neighbors):
                neighbor_hub_id = sorted_neighbors[i]
                
                # A. Travel Time to Neighbor (Real-time estimated travel time)
                travel_time_to_nk = 1.0  # Default normalized value
                try:
                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    target_ingoing_edges = self.hub_ingoing_edges.get(neighbor_hub_id, [])
                    
                    if current_edge and target_ingoing_edges:
                        # Use the first ingoing edge as a representative target
                        target_edge = target_ingoing_edges[0]
                        route_info = traci.simulation.findRoute(current_edge, target_edge)
                        # Normalize by a reasonable max travel time (e.g., 100s)
                        travel_time_to_nk = min(1.0, route_info.travelTime / 100.0)
                    else:
                        # Fallback to static travel time if dynamic routing failed
                        static_tt = self.hub_network_distances.get(hub_id, {}).get(neighbor_hub_id, 100.0)
                        travel_time_to_nk = min(1.0, static_tt / 100.0)
                except traci.exceptions.TraCIException:
                    static_tt = self.hub_network_distances.get(hub_id, {}).get(neighbor_hub_id, 100.0)
                    travel_time_to_nk = min(1.0, static_tt / 100.0)

                # B. Neighbor Incoming Congestion Ratio (Predicts congestion at approach)
                nk_incoming_congestion = self.get_hub_incoming_congestion_ratio(neighbor_hub_id)
                
                # C. Distance from Neighbor to Destination (Heuristic for future cost)
                dist_nk_to_dest = 1.0  # Default normalized value
                max_dist = 1000.0  # Heuristic max network distance for normalization
                if (neighbor_hub_id in self.hub_network_distances and 
                    dest_hub_id in self.hub_network_distances[neighbor_hub_id]):
                    distance = self.hub_network_distances[neighbor_hub_id][dest_hub_id]
                    dist_nk_to_dest = min(1.0, distance / max_dist)
                
                padded_neighbor_features.extend([travel_time_to_nk, nk_incoming_congestion, dist_nk_to_dest])
            else:
                # Padding for non-existent neighbors (use -1.0 to indicate invalid/padded features)
                padded_neighbor_features.extend([-1.0, -1.0, -1.0])

        # 4. Combine all features into the final state vector
        state_vector = np.concatenate([
            dest_loc_embedding,  # Goal representation
            np.array([current_hub_vicinity_speed, current_hub_outgoing_congestion]),  # Current hub conditions
            np.array(padded_neighbor_features)  # Padded neighbor features
        ]).astype(np.float32)

        return state_vector, action_mask
    
    def get_hub_vicinity_speed(self, hub_id, radius=1300):
        """
        Calculates the average normalized speed of vehicles in a hub's vicinity.
        
        Args:
            hub_id: The ID of the hub.
            radius: The radius (in meters) to define the vicinity.
            
        Returns:
            A float for the normalized average speed (0 to 1).
        """
        hub_pos = self.hub_coordinates.get(hub_id)
        if not hub_pos: return 0.0

        total_norm_speed = 0.0
        vehicle_count = 0
        radius_sq = radius * radius

        try:
            for vehicle_id in traci.vehicle.getIDList():
                veh_pos = traci.vehicle.getPosition(vehicle_id)
                if (veh_pos[0] - hub_pos[0])**2 + (veh_pos[1] - hub_pos[1])**2 < radius_sq:
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    allowed_speed = traci.vehicle.getAllowedSpeed(vehicle_id)
                    norm_speed = speed / max(1.0, allowed_speed)
                    total_norm_speed += norm_speed
                    vehicle_count += 1
        except traci.exceptions.TraCIException:
            return 0.0 # Return default on simulation error

        return total_norm_speed / vehicle_count if vehicle_count > 0 else 0.0

    def get_hub_processing_rate(self, hub_id):
        """
        Calculates a normalized processing rate for the hub.
        This is a proxy for throughput, measured as successfully routed vehicles over time.
        
        Args:
            hub_id: The ID of the hub.
            
        Returns:
            A float for the normalized processing rate.
        """
        agent = self.hub_agents.get(hub_id)
        if not agent or self.current_time == 0:
            return 0.0
        
        # Rate = vehicles per second
        rate = agent.num_vehicles_routed / self.current_time
        # Normalize based on a heuristic maximum rate (e.g., 0.5 vehicles/sec is high)
        return min(1.0, rate / 0.5)
    
    def get_hub_outgoing_congestion_ratio(self, hub_id):
        """
        Calculates the average travel time ratio (current/free-flow) on edges leaving the hub.
        A value > 1.0 indicates it's taking longer than normal for traffic to disperse from the hub.
        
        Args:
            hub_id: The ID of the hub.
            
        Returns:
            A float for the normalized outgoing congestion ratio.
        """
        if hub_id not in self.hubs:
            return 0.0
        
        hub_node = self.hubs[hub_id]
        outgoing_edges = []
        
        # Get outgoing SUMO edges from this hub
        for successor in self.sumo_graph.successors(hub_node):
            edge_data = self.sumo_graph.get_edge_data(hub_node, successor)
            if edge_data and 'edge' in edge_data:
                edge_id = edge_data['edge'].id
                if edge_id:
                    outgoing_edges.append(edge_id)
        
        if not outgoing_edges:
            return 0.0
        
        total_congestion_ratio = 0.0
        valid_edges = 0
        
        for edge_id in outgoing_edges:
            try:
                edge_obj = self._get_edge_by_id(edge_id)
                if not edge_obj: 
                    continue
                current_tt = traci.edge.getTraveltime(edge_id)
                free_flow_tt = edge_obj.length / max(0.1, edge_obj.flow_speed)
                congestion_ratio = current_tt / max(1.0, free_flow_tt)
                total_congestion_ratio += congestion_ratio
                valid_edges += 1
            except:
                continue
        
        if valid_edges == 0:
            return 0.0
        
        avg_congestion_ratio = total_congestion_ratio / valid_edges
        # Normalize to [0,1]: values around 1.0 are normal, higher values indicate congestion
        return min(1.0, (avg_congestion_ratio - 1.0) / 4.0)  # Cap at 5x congestion ratio
    
    def get_hub_incoming_congestion_ratio(self, hub_id):
        """
        Calculates the average travel time ratio on edges entering the hub.
        This predicts the congestion a vehicle will face as it approaches this hub.
        
        Args:
            hub_id: The ID of the hub.
            
        Returns:
            A float for the normalized incoming congestion ratio.
        """
        incoming_edges = self.hub_ingoing_edges.get(hub_id, [])
        
        if not incoming_edges:
            return 0.0
        
        total_congestion_ratio = 0.0
        valid_edges = 0
        
        for edge_id in incoming_edges:
            try:
                edge_obj = self._get_edge_by_id(edge_id)
                if not edge_obj:
                    continue
                current_tt = traci.edge.getTraveltime(edge_id)
                free_flow_tt = edge_obj.length / max(0.1, edge_obj.flow_speed)
                congestion_ratio = current_tt / max(1.0, free_flow_tt)
                total_congestion_ratio += congestion_ratio
                valid_edges += 1
            except:
                continue
        
        if valid_edges == 0:
            return 0.0
        
        avg_congestion_ratio = total_congestion_ratio / valid_edges
        # Normalize to [0,1]: values around 1.0 are normal, higher values indicate congestion
        return min(1.0, (avg_congestion_ratio - 1.0) / 4.0)  # Cap at 5x congestion ratio
    
    def get_completion_throughput_ratio(self):
        """
        Calculates the completion throughput ratio as completed vehicles / started vehicles.
        This is a direct measure of system efficiency.
        
        Returns:
            A float for the completion throughput ratio, normalized to [0,1].
        """
        if self.total_journey_starts == 0:
            return 0.0
        return self.total_vehicles_completed / self.total_journey_starts

    def get_avg_completed_trip_inefficiency(self):
        """
        Returns the average trip inefficiency from the list of completed trips.
        
        Returns:
            A float representing the average inefficiency (1.0 is perfect).
        """
        if not self.completed_trip_inefficiencies:
            return 1.0 # Default to perfect efficiency if no data
        
        return np.mean(self.completed_trip_inefficiencies)


    def _get_shortest_path(self, from_node, to_node):
        """Get shortest path between two nodes using the SUMO graph."""
        if not nx.has_path(self.sumo_graph, from_node, to_node):
            return []
        # Use weight='weight' which should correspond to edge length or time
        return nx.shortest_path(self.sumo_graph, from_node, to_node, weight='weight')
    
    def get_available_actions_for_vehicle(self, hub_id, vehicle_id):
        """
        Returns a mask of available actions for a vehicle, masking out U-turn actions.
        Now masks out the previous hub the vehicle came from to prevent hub-level U-turns.
        
        Args:
            hub_id: The ID of the hub where the vehicle is waiting.
            vehicle_id: The ID of the vehicle.
            
        Returns:
            A boolean array where True indicates the action is available.
        """
        if vehicle_id not in self.vehicles or hub_id not in self.hub_agents:
            return None
            
        # Get hub agent to access neighbors
        agent = self.hub_agents[hub_id]
        action_space_size = agent.action_space_size
        action_mask = np.ones(action_space_size, dtype=bool)  # Default: all actions available
        
        # Get the previous hub the vehicle came from (if any)
        vehicle = self.vehicles[vehicle_id]
        previous_hub_id = vehicle.get('previous_hub')
        
        # If this is not the first hub or we don't have previous hub info, all actions are valid
        if previous_hub_id is None:
            return action_mask
            
        # Mask out the previous hub to prevent U-turns
        try:
            # Find the index of the previous hub in the sorted neighbors list
            if previous_hub_id in agent.neighbors:
                previous_hub_idx = agent.neighbors.index(previous_hub_id)
                action_mask[previous_hub_idx] = False
        except:
            pass
            
        return action_mask

    # In file: hubroutingenvnew.py

    def reset(self, seed=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
            # Note: PyTorch seeding should be handled externally if needed

        try: traci.close()
        except: pass

        self._init_sumo() # Restart SUMO

        self.vehicles = {}
        self.active_hubs = set()
        self.total_travel_time = 0
        self.total_vehicles_completed = 0
        self.total_journey_starts = 0
        self.waiting_vehicles_per_hub = {hub_id: 0 for hub_id in self.hub_agents}
        self.routed_to_final_vehicles = set()
        self.completed_vehicle_ids = set() # MODIFICATION: Reset the set for the new episode
        self.current_time = 0
        self.completed_trip_inefficiencies = []
        
        # Reset the pending experiences tracker
        self.pending_experiences = {}

        for agent in self.hub_agents.values():
            agent.num_vehicles_routed = 0
            agent.total_routing_time = 0

        # Run one step to initialize simulation state
        traci.simulationStep()
        self.current_time += self.simulation_step
        self._check_vehicles_at_hubs() # Process any initially loaded vehicles

        # Get initial states for waiting vehicles
        initial_states = {}
        for hub_id in self.active_hubs:
            initial_states[hub_id] = {}
            waiting_vehicles_at_hub = [
                v_id for v_id, v_data in self.vehicles.items()
                if v_data.get('current_hub') == hub_id and 
                not v_data.get('has_next_hub') and 
                v_data.get('waiting_at_hub', False)
            ]
            for vehicle_id in waiting_vehicles_at_hub:
                state_result = self._get_dqr_state_for_vehicle(hub_id, vehicle_id)
                if state_result is not None:
                    state_vec, action_mask = state_result
                    initial_states[hub_id][vehicle_id] = {
                        'state': state_vec,
                        'action_mask': action_mask
                    }

        # The 'observation' is now a dict of states for waiting vehicles
        return initial_states, {} # Return state dict and empty info

    def step(self, actions):
        """
        Take one environment step based on DQR actions.

        Args:
            actions: Dictionary of {hub_id: {vehicle_id: {'action': action_index, 'state': state_vector}}}
                     where action_index selects a target neighboring hub.

        Returns:
            next_states, reward, terminated, truncated, info
            - next_states: Dict {hub_id: {vehicle_id: state_vector}} for vehicles now waiting
            - reward: Single float reward (using global signal for now, hub_rewards in info)
            - terminated: Boolean, True if simulation ended normally.
            - truncated: Boolean, False (no truncation condition implemented).
            - info: Dictionary with metrics and completed experiences for RL buffer.
        """
        # Initialize metrics for this step
        step_routing_successes = 0
        step_total_routing_time = 0
        step_completed_vehicles = 0
        step_total_travel_time = 0
        step_journey_starts = 0

        # Initialize dictionary to collect completed experiences
        completed_experiences = {}  # {hub_id: [experience_tuples]}

        # 1. Process agent actions
        if actions:
            routing_results = self._process_actions(actions)
            step_routing_successes = routing_results['routing_successes']
            step_total_routing_time = routing_results['total_routing_time']

        # 2. Run simulation step
        try:
            traci.simulationStep()
            self.current_time += self.simulation_step
            sim_ok = True
        except traci.exceptions.TraCIException as e:
             # print(f"TraCI Error during simulation step: {e}. Assuming simulation ended.")
             sim_ok = False

        # 3. Update vehicle states, check arrivals, handle completions
        if sim_ok:
            vehicle_results = self._check_vehicles_at_hubs()
            step_completed_vehicles = vehicle_results['completed_vehicles']
            step_total_travel_time = vehicle_results['total_travel_time']
            step_journey_starts = vehicle_results['journey_starts']
            completed_vehicle_ids = vehicle_results.get('completed_vehicle_ids', [])
            
            # Get completed experiences from vehicle arrivals
            new_completed_experiences = vehicle_results.get('completed_experiences', {})
            
            # Merge with existing completed experiences
            for hub_id, experiences in new_completed_experiences.items():
                if hub_id not in completed_experiences:
                    completed_experiences[hub_id] = []
                completed_experiences[hub_id].extend(experiences)
        else:
             # If sim failed, process completions for vehicles already tracked
             completed_vehicle_ids = list(self.routed_to_final_vehicles)
             for v_id in completed_vehicle_ids:
                 if v_id in self.vehicles:
                      vehicle = self.vehicles[v_id]
                      if 'start_time' in vehicle:
                           travel_time = self.current_time - vehicle['start_time'] # Use current time
                           self.total_travel_time += travel_time
                           step_total_travel_time += travel_time
                      self.total_vehicles_completed += 1
                      step_completed_vehicles += 1
                      self.vehicles.pop(v_id, None) # Clean up

             self.routed_to_final_vehicles.clear()
             step_journey_starts = 0

        # 4. Calculate hub-specific rewards based on actual travel times from completed experiences
        hub_rewards = {}
        
        # Initialize rewards for all hubs (so none are missed)
        for hub_id in self.hub_agents:
            hub_rewards[hub_id] = 0.0
            
        # Sum rewards for each hub from completed experiences
        for hub_id, experiences in completed_experiences.items():
            total_reward = 0
            for exp in experiences:
                total_reward += exp['reward']
            
            if hub_id in hub_rewards:
                hub_rewards[hub_id] = total_reward
                
            # print(f"Hub {hub_id} reward: {hub_rewards[hub_id]:.2f} (based on {len(experiences)} completed experiences)")

        # Calculate global reward as sum of all hub rewards
        global_reward = sum(hub_rewards.values())

        # 5. Check termination condition
        terminated = False
        if sim_ok:
             try:
                  terminated = traci.simulation.getMinExpectedNumber() <= 0
             except traci.exceptions.TraCIException:
                  # print("TraCI Error checking termination. Assuming terminated.")
                  terminated = True
        else:
             terminated = True # Simulation connection lost

        # 6. Construct next state observation
        next_states = {}
        for hub_id in self.active_hubs:
            next_states[hub_id] = {}
            waiting_vehicles_at_hub = [
                v_id for v_id, v_data in self.vehicles.items()
                if v_data.get('current_hub') == hub_id and 
                not v_data.get('has_next_hub') and 
                v_data.get('waiting_at_hub', False)
            ]
            for vehicle_id in waiting_vehicles_at_hub:
                state_result = self._get_dqr_state_for_vehicle(hub_id, vehicle_id)
                if state_result is not None:
                    state_vec, action_mask = state_result
                    next_states[hub_id][vehicle_id] = {
                        'state': state_vec,
                        'action_mask': action_mask
                    }

        # 7. Collect info
        info = {
            'num_vehicles': len(traci.vehicle.getIDList()) if sim_ok else 0,
            'num_waiting_vehicles': sum(self.waiting_vehicles_per_hub.values()),
            'active_hubs': len(self.active_hubs),
            'step_total_travel_time': step_total_travel_time,
            'step_vehicles_completed': step_completed_vehicles,
            'step_routing_successes': step_routing_successes,
            'step_total_routing_time': step_total_routing_time,
            'step_journey_starts': step_journey_starts,
            'hub_rewards': hub_rewards,  # Contains negative travel time rewards
            'completed_experiences': completed_experiences,  # For populating experience buffer
            'total_vehicles_completed': self.total_vehicles_completed,
            'total_journey_starts': self.total_journey_starts,
            'current_time': self.current_time,
            'pending_experiences_count': {hub_id: len(vehicles) for hub_id, vehicles in self.pending_experiences.items()}
        }

        if terminated:
            #print("total travel time", self.total_travel_time)
            #print("total vehicles completed", self.total_vehicles_completed)
            avg_travel_time = self.total_travel_time / max(1, self.total_vehicles_completed)
            info['final_average_travel_time'] = avg_travel_time
            # print(f"Simulation ended. Total completed: {self.total_vehicles_completed}, Avg Time: {avg_travel_time:.2f}")

        # Return structure: next_states_dict, global_reward, terminated, truncated, info_dict
        return next_states, global_reward, terminated, False, info
    

    def _process_actions(self, actions):
        """
        Process actions based on the chosen target hub.
        MODIFIED:
        - If the next hub is the final destination, routes directly to the original trip's final edge.
        - If routing to an intermediate hub, chooses the least congested ingoing edge.
        """
        routing_successes = 0
        total_routing_time = 0

        for hub_id, vehicle_actions in actions.items():
            if hub_id not in self.hub_agents: continue

            agent = self.hub_agents[hub_id]

            for vehicle_id, action_data in vehicle_actions.items():
                if vehicle_id not in self.vehicles: continue

                action_index = action_data.get('action')
                state_x = action_data.get('state')

                if action_index is None or state_x is None:
                    print(f"Warning: Missing action_index or state for vehicle {vehicle_id}. Skipping.")
                    continue

                vehicle = self.vehicles[vehicle_id]
                if vehicle.get('current_hub') != hub_id or vehicle.get('has_next_hub'): continue

                next_hub_id = agent.get_target_hub_for_action(action_index)

                if next_hub_id is None:
                    print(f"Warning: Invalid action index {action_index} for hub {hub_id}. Skipping.")
                    continue

                try:
                    if vehicle_id not in traci.vehicle.getIDList():
                        print(f"Warning: Vehicle {vehicle_id} not in TraCI. Skipping.")
                        continue

                    # Store data for experience replay before attempting routing
                    vehicle['pending_experience_start'] = {
                        'state': state_x,
                        'action': action_index,
                        'origin_hub': hub_id,
                        'hop_start_time': self.current_time
                    }

                    # --- MODIFICATION: Check if next hub is the destination hub ---
                    if next_hub_id == vehicle.get('destination_hub'):
                        final_edge = vehicle.get('final_edge')
                        if final_edge:
                            try:
                                traci.vehicle.changeTarget(vehicle_id, final_edge)
                                vehicle['next_hub'] = next_hub_id
                                vehicle['has_next_hub'] = True
                                vehicle['previous_hub'] = hub_id
                                vehicle['routed_to_final'] = True
                                self.routed_to_final_vehicles.add(vehicle_id)
                                routing_successes += 1
                                # No longer waiting, so decrement count
                                self.waiting_vehicles_per_hub[hub_id] = max(0, self.waiting_vehicles_per_hub[hub_id] - 1)
                                vehicle['waiting_at_hub'] = False
                                continue  # Skip to next vehicle
                            except traci.exceptions.TraCIException as e:
                                print(f"Error routing vehicle {vehicle_id} to final edge {final_edge}: {e}")
                                # If direct routing fails, clear pending experience and let it be re-processed
                                vehicle.pop('pending_experience_start', None)
                                continue

                    # --- MODIFICATION: If not final hub, find least congested ingoing edge ---
                    target_ingoing_edges = self.hub_ingoing_edges.get(next_hub_id, [])
                    if not target_ingoing_edges:
                        print(f"Error: No ingoing edges for hub {next_hub_id}. Cannot route vehicle {vehicle_id}.")
                        vehicle.pop('pending_experience_start', None)
                        continue

                    least_congested_edge = None
                    min_vehicle_count = float('inf')
                    
                    for edge_id in target_ingoing_edges:
                        try:
                            vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                            if vehicle_count < min_vehicle_count:
                                min_vehicle_count = vehicle_count
                                least_congested_edge = edge_id
                        except traci.exceptions.TraCIException:
                            continue
                    
                    target_edge = least_congested_edge if least_congested_edge else target_ingoing_edges[0]
                    
                    # Route to the selected intermediate edge
                    try:
                        traci.vehicle.changeTarget(vehicle_id, target_edge)
                        vehicle['next_hub'] = next_hub_id
                        vehicle['has_next_hub'] = True
                        vehicle['previous_hub'] = hub_id
                        vehicle['waiting_at_hub'] = False
                        routing_successes += 1
                        self.waiting_vehicles_per_hub[hub_id] = max(0, self.waiting_vehicles_per_hub[hub_id] - 1)

                    except traci.exceptions.TraCIException as e:
                        print(f"Error routing vehicle {vehicle_id} to intermediate edge {target_edge}: {e}")
                        vehicle.pop('pending_experience_start', None)
                        continue

                except traci.exceptions.TraCIException as e:
                    print(f"Error processing action for vehicle {vehicle_id}: {e}")
                    if 'pending_experience_start' in vehicle:
                        vehicle.pop('pending_experience_start', None)

        self.active_hubs = {hub for hub, count in self.waiting_vehicles_per_hub.items() if count > 0}
        return {'routing_successes': routing_successes, 'total_routing_time': total_routing_time}


    # In file: hubroutingenvnew.py

    def _check_vehicles_at_hubs(self):
        """
        Check vehicle arrivals, handle completions, and find new waiting vehicles.
        MODIFIED: 
        - Completion is now ONLY recognized when a vehicle is on its final edge.
        - Disappeared vehicles are no longer counted as successful completions to prevent exploits.
        - Added a minimum deadline for new vehicles.
        - PREVENTS RE-INITIALIZATION OF COMPLETED VEHICLES.
        """
        step_results = {
            'completed_vehicles': 0,
            'total_travel_time': 0,
            'journey_starts': 0,
            'completed_experiences': {}
            # MODIFICATION: 'completed_vehicle_ids' list is removed from step_results as it's redundant
        }

        try:
            current_vehicle_ids = set(traci.vehicle.getIDList())
        except traci.exceptions.TraCIException:
            current_vehicle_ids = set()

        vehicles_to_remove = []

        # --- 1. Process vehicles that have disappeared from the simulation ---
        for vehicle_id, vehicle in list(self.vehicles.items()):
            if vehicle_id not in current_vehicle_ids:
                vehicles_to_remove.append(vehicle_id)

        # --- 2. Process vehicles still present in the simulation ---
        for vehicle_id in current_vehicle_ids:
            # --- A. Handle New Vehicles ---
            if vehicle_id not in self.vehicles:
                
                # MODIFICATION: CRITICAL FIX - Check if the vehicle has already completed its journey in this episode.
                if vehicle_id in self.completed_vehicle_ids:
                    continue # If so, ignore it completely.
                    
                try:
                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    if current_edge.startswith(':'): continue

                    original_route = traci.vehicle.getRoute(vehicle_id)
                    if not original_route: continue

                    final_edge_id = original_route[-1]
                    final_edge_obj = self._get_edge_by_id(final_edge_id)
                    if not final_edge_obj or not final_edge_obj.to_id: continue

                    current_edge_obj = self._get_edge_by_id(current_edge)
                    start_node = current_edge_obj.from_id if current_edge_obj else None
                    final_node = final_edge_obj.to_id

                    if not start_node or not final_node: continue

                    vehicle_pos = traci.vehicle.getPosition(vehicle_id)
                    nearest_hub_id = min(self.hub_coordinates, 
                                        key=lambda hid: (vehicle_pos[0] - self.hub_coordinates[hid][0])**2 + (vehicle_pos[1] - self.hub_coordinates[hid][1])**2)
                    
                    dest_hub_id = None
                    if final_node in self.node_positions:
                        final_pos = self.node_positions[final_node]
                        dest_hub_id = min(self.hub_coordinates, 
                                        key=lambda hid: (final_pos[0] - self.hub_coordinates[hid][0])**2 + (final_pos[1] - self.hub_coordinates[hid][1])**2)

                    if not nearest_hub_id or not dest_hub_id: continue

                    shortest_path_time = self.all_pairs_shortest_path.get(start_node, {}).get(final_node, 0)
                    deadline = self.current_time + max(20000, 5 * shortest_path_time) 

                    target_edge = final_edge_id if nearest_hub_id == dest_hub_id else self.hub_ingoing_edges.get(nearest_hub_id, [None])[0]
                    if not target_edge: continue
                    
                    traci.vehicle.changeTarget(vehicle_id, target_edge)
                    self.vehicles[vehicle_id] = {
                        'current_hub': nearest_hub_id, 'destination_hub': dest_hub_id,
                        'next_hub': None, 'has_next_hub': False,
                        'start_time': self.current_time, 'arrival_time': None,
                        'routing_history': [], 'routed_to_final': nearest_hub_id == dest_hub_id,
                        'final_edge': final_edge_id, 'waiting_at_hub': False,
                        'deadline': deadline, 'shortest_path_time': shortest_path_time
                    }
                    if nearest_hub_id == dest_hub_id:
                        self.routed_to_final_vehicles.add(vehicle_id)
                    step_results['journey_starts'] += 1
                    self.total_journey_starts += 1
                except (traci.exceptions.TraCIException, KeyError, AttributeError):
                    continue

            # --- B. Handle Existing Vehicles ---
            else:
                vehicle = self.vehicles[vehicle_id]
                try:
                    if 'deadline' in vehicle and self.current_time > vehicle['deadline']:
                        vehicles_to_remove.append(vehicle_id)
                        continue

                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    is_waiting = vehicle.get('waiting_at_hub', False)
                    has_next = vehicle.get('has_next_hub', False)
                    is_final_routed = vehicle.get('routed_to_final', False)
                    
                    if is_final_routed and current_edge == vehicle.get('final_edge'):
                        travel_time = self.current_time - vehicle['start_time']
                        step_results['total_travel_time'] += travel_time
                        self.total_travel_time += travel_time
                        step_results['completed_vehicles'] += 1
                        self.total_vehicles_completed += 1
                        
                        # MODIFICATION: Add the vehicle ID to the authoritative set of completed vehicles.
                        self.completed_vehicle_ids.add(vehicle_id)
                        
                        shortest_time = vehicle.get('shortest_path_time', 0)
                        if shortest_time > 0:
                            inefficiency = travel_time / shortest_time
                            self.completed_trip_inefficiencies.append(inefficiency)
                        
                        vehicles_to_remove.append(vehicle_id)
                        self.routed_to_final_vehicles.discard(vehicle_id)
                        continue

                    arriving_hub_id = None
                    if not is_waiting and not is_final_routed:
                        hub_to_check = vehicle.get('next_hub') if has_next else vehicle.get('current_hub')
                        
                        on_ingoing_edge = False
                        near_hub = False
                        
                        if hub_to_check and hub_to_check in self.hub_ingoing_edges:
                            on_ingoing_edge = current_edge in self.hub_ingoing_edges[hub_to_check]
                        
                        if hub_to_check and hub_to_check in self.hub_coordinates:
                            try:
                                vehicle_pos = traci.vehicle.getPosition(vehicle_id)
                                hub_pos = self.hub_coordinates[hub_to_check]
                                dist_sq = (vehicle_pos[0] - hub_pos[0])**2 + (vehicle_pos[1] - hub_pos[1])**2
                                if dist_sq <= 2250000: # 1500m radius
                                    near_hub = True
                            except traci.exceptions.TraCIException:
                                pass

                        if on_ingoing_edge or near_hub:
                            arriving_hub_id = hub_to_check

                    if arriving_hub_id:
                        if 'pending_experience_start' in vehicle:
                            exp_start_data = vehicle.pop('pending_experience_start')
                            hop_time_t_xy = self.current_time - exp_start_data['hop_start_time']
                            state_mask_result = self._get_dqr_state_for_vehicle(arriving_hub_id, vehicle_id)
                            if state_mask_result:
                                next_state_sy, next_state_action_mask = state_mask_result
                                experience = {
                                    'state': exp_start_data['state'], 'action': exp_start_data['action'],
                                    'travel_time': hop_time_t_xy, 'next_state': next_state_sy,
                                    'destination_hub_id': vehicle['destination_hub'], 'next_hub': arriving_hub_id,
                                    'next_state_action_mask': next_state_action_mask, 'reward': -hop_time_t_xy
                                }
                                if exp_start_data['origin_hub'] not in step_results['completed_experiences']:
                                    step_results['completed_experiences'][exp_start_data['origin_hub']] = []
                                step_results['completed_experiences'][exp_start_data['origin_hub']].append(experience)

                        vehicle['current_hub'] = arriving_hub_id
                        vehicle['next_hub'] = None
                        vehicle['has_next_hub'] = False
                        vehicle['arrival_time'] = self.current_time

                        if arriving_hub_id == vehicle.get('destination_hub'):
                            final_edge = vehicle.get('final_edge')
                            if final_edge:
                                traci.vehicle.changeTarget(vehicle_id, final_edge)
                                vehicle['routed_to_final'] = True
                                vehicle['waiting_at_hub'] = False
                                self.routed_to_final_vehicles.add(vehicle_id)
                        else:
                            vehicle['waiting_at_hub'] = True
                            self.waiting_vehicles_per_hub[arriving_hub_id] = self.waiting_vehicles_per_hub.get(arriving_hub_id, 0) + 1
                            self.active_hubs.add(arriving_hub_id)

                except traci.exceptions.TraCIException:
                    if vehicle_id not in current_vehicle_ids:
                        vehicles_to_remove.append(vehicle_id)

        # --- 3. Clean up removed vehicles from internal tracking ---
        for v_id in vehicles_to_remove:
            if v_id in self.vehicles:
                hub_id = self.vehicles[v_id].get('current_hub')
                if hub_id and self.vehicles[v_id].get('waiting_at_hub'):
                    self.waiting_vehicles_per_hub[hub_id] = max(0, self.waiting_vehicles_per_hub[hub_id] - 1)
                self.vehicles.pop(v_id, None)
            
            self.routed_to_final_vehicles.discard(v_id)

        self.active_hubs = {hub for hub, count in self.waiting_vehicles_per_hub.items() if count > 0}
        return step_results
    
    def _get_avg_outgoing_congestion(self, hub_id):
        """
        Calculates the average path congestion for departures from a given hub
        to its own neighbors, indicating if the hub is a future bottleneck.
        
        Args:
            hub_id: The ID of the hub to assess.
            
        Returns:
            A float representing the normalized average congestion (0 to 1).
        """
        if hub_id not in self.hub_graph:
            return 0.0

        total_congestion_ratio = 0.0
        num_valid_neighbors = 0
        
        source_node = self.hubs[hub_id]
        neighbors = list(self.hub_graph.successors(hub_id))

        if not neighbors:
            return 0.0

        for neighbor_id in neighbors:
            target_node = self.hubs[neighbor_id]
            try:
                # Use SUMO's findRoute to get a dynamic travel time estimate
                source_outgoing_edges = [e.getID() for e in self.sumo_net.getNode(source_node).getOutgoing()]
                target_ingoing_edges = [e.getID() for e in self.sumo_net.getNode(target_node).getIncoming()]

                if not source_outgoing_edges or not target_ingoing_edges:
                    continue

                route_info = traci.simulation.findRoute(source_outgoing_edges[0], target_ingoing_edges[0])
                if route_info.travelTime > 0:
                    # Get static (free-flow) travel time for comparison
                    static_travel_time = self.hub_network_distances.get(hub_id, {}).get(neighbor_id, route_info.travelTime)
                    congestion_ratio = route_info.travelTime / max(1.0, static_travel_time)
                    total_congestion_ratio += congestion_ratio
                    num_valid_neighbors += 1
            except traci.exceptions.TraCIException:
                continue # Skip if no route exists

        if num_valid_neighbors == 0:
            return 0.0
        
        # Average the ratios and normalize (clamping at a high value like 5 to keep it reasonable)
        avg_congestion = total_congestion_ratio / num_valid_neighbors
        return min(1.0, (avg_congestion - 1.0) / 4.0) # Normalize so 1.0 is no congestion, >1 is congestion

    def get_active_hubs(self):
        """Get the set of hubs that currently have vehicles waiting."""
        # Recompute based on current waiting counts
        self.active_hubs = {hub_id for hub_id, count in self.waiting_vehicles_per_hub.items() if count > 0}
        return self.active_hubs

    def close(self):
        """Close the environment and clean up resources."""
        print("Closing HubRoutingEnv and cleaning up resources...")
        
        # Close TRACI connection if active
        try:
            if traci.isConnected():
                traci.close()
                print("SUMO connection closed successfully.")
        except Exception as e:
            print(f"Error closing TRACI connection: {e}")
        
        # Clean up temporary files
        removed_files = 0
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try: 
                    os.remove(temp_file)
                    removed_files += 1
                except OSError as e: 
                    print(f"Error removing temp file {temp_file}: {e}")
        
        if removed_files > 0:
            print(f"Cleaned up {removed_files} temporary files.")
        
        # Reset critical attributes
        self.vehicles = {}
        self.active_hubs = set()
        self.waiting_vehicles_per_hub = {hub_id: 0 for hub_id in self.hub_agents}
        self.routed_to_final_vehicles = set()
        
        print("HubRoutingEnv closed successfully.")

    def visualize_hubs(self, save_path=None, show_connections=True, hub_size=150, hub_color='red', 
                     edge_color='blue', edge_width=1.5, background_color='lightgray',
                     node_color='gray', node_size=20, figsize=(12, 10), hub_radius=1400):
        """
        Visualize hub locations on the SUMO network map.
        
        Args:
            save_path (str, optional): Path to save the visualization image. If None, the image is not saved.
            show_connections (bool): Whether to show connections between hubs.
            hub_size (int): Size of hub markers.
            hub_color (str): Color of hub markers.
            edge_color (str): Color of edges between hubs.
            edge_width (float): Width of edges between hubs.
            background_color (str): Color of background.
            node_color (str): Color of regular nodes.
            node_size (int): Size of regular nodes.
            figsize (tuple): Figure size in inches.
            hub_radius (float): Radius of circles drawn around each hub.
            
        Returns:
            matplotlib.figure.Figure: The figure object containing the visualization.
        """
        if not self.hub_coordinates or len(self.hub_coordinates) == 0:
            print("Error: No hub coordinates available for visualization.")
            return None
            
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Circle
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor(background_color)
        
        # Plot all network nodes as small dots for context
        if self.node_positions:
            node_x = [pos[0] for pos in self.node_positions.values()]
            node_y = [pos[1] for pos in self.node_positions.values()]
            ax.scatter(node_x, node_y, color=node_color, s=node_size, alpha=0.5, zorder=1)
        
        # Plot network edges as lines for context
        lines = []
        for edge in self.network_model.edges.values():
            if hasattr(edge, 'shape') and hasattr(edge.shape, 'points') and edge.shape.points:
                points = edge.shape.points
                lines.append(points)
        
        if lines:
            lc = LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.6, zorder=2)
            ax.add_collection(lc)
        
        # Plot hub connections if requested
        if show_connections and self.hub_graph:
            hub_connections = []
            for u, v in self.hub_graph.edges():
                if u in self.hub_coordinates and v in self.hub_coordinates:
                    start_pos = self.hub_coordinates[u]
                    end_pos = self.hub_coordinates[v]
                    hub_connections.append([start_pos, end_pos])
            
            if hub_connections:
                lc_hubs = LineCollection(hub_connections, colors=edge_color, linewidths=edge_width, 
                                        alpha=0.7, zorder=3)
                ax.add_collection(lc_hubs)
        
        # Add circles around each hub
        for hub_id, pos in self.hub_coordinates.items():
            circle = Circle((pos[0], pos[1]), hub_radius, fill=False, edgecolor=hub_color, 
                          linewidth=1.5, alpha=0.6, zorder=3)
            ax.add_patch(circle)
        
        # Plot hubs as larger circles with labels
        hub_x = [pos[0] for pos in self.hub_coordinates.values()]
        hub_y = [pos[1] for pos in self.hub_coordinates.values()]
        
        # Plot hub points
        ax.scatter(hub_x, hub_y, color=hub_color, s=hub_size, marker='o', 
                  edgecolor='black', linewidth=1.5, zorder=4)
        
        # Add hub labels
        for hub_id, pos in self.hub_coordinates.items():
            ax.text(pos[0], pos[1], hub_id, fontsize=10, ha='center', va='center', 
                   color='white', fontweight='bold', zorder=5)
        
        # Set axis limits to show the complete network instead of just hubs
        if self.node_positions:
            # Use all node positions to determine the complete network bounds
            all_x = [pos[0] for pos in self.node_positions.values()]
            all_y = [pos[1] for pos in self.node_positions.values()]
            
            if all_x and all_y:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                
                # Add small padding (5% of range) to ensure everything is visible
                padding_x = (max_x - min_x) * 0.05
                padding_y = (max_y - min_y) * 0.05
                
                ax.set_xlim(min_x - padding_x, max_x + padding_x)
                ax.set_ylim(min_y - padding_y, max_y + padding_y)
        elif hub_x and hub_y:
            # Fallback to hub-based bounds if node positions unavailable
            min_x, max_x = min(hub_x), max(hub_x)
            min_y, max_y = min(hub_y), max(hub_y)
            
            # Add padding (10% of range)
            padding_x = (max_x - min_x) * 0.1
            padding_y = (max_y - min_y) * 0.1
            
            ax.set_xlim(min_x - padding_x, max_x + padding_x)
            ax.set_ylim(min_y - padding_y, max_y + padding_y)
        
        # Ensure aspect ratio is equal for accurate distance representation
        ax.set_aspect('equal', adjustable='box')
        
        # Add title and labels
        plt.title(f'Hub Locations on SUMO Network (Total: {len(self.hub_coordinates)})', fontsize=14)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Add a legend
        hub_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=hub_color, 
                               markersize=10, label='Hub')
        node_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_color, 
                                markersize=5, label='Network Node')
        edge_line = plt.Line2D([0], [0], color=edge_color, linewidth=edge_width, 
                              label='Hub Connection')
        hub_circle = plt.Line2D([0], [0], color=hub_color, linewidth=1.5, 
                               label=f'Hub Radius ({hub_radius}m)')
        
        plt.legend(handles=[hub_marker, node_marker, edge_line, hub_circle], loc='best')
        
        plt.tight_layout()
        
        # Save if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        return fig

    def log_local_state(self, local_state, hub_id, vehicle_id):
        """Logs the components of the local state vector with labels."""
        vehicle_info = self.vehicles.get(vehicle_id, {})
        dest_hub_id = vehicle_info.get('destination_hub', 'Unknown')
        print(f"\n--- Logging Local State for Veh '{vehicle_id}' at Hub '{hub_id}' (Destination: {dest_hub_id}) ---")
        if local_state is None or local_state.size == 0:
            print("Local state is empty or None.")
            return

        z_dim = self.z_order_embedding_dim
        neighbor_features_dim = self.max_neighbors * 3
        
        expected_dim = z_dim + neighbor_features_dim
        if local_state.shape[0] != expected_dim:
            print(f"ERROR: Local state dimension mismatch. Expected {expected_dim}, got {local_state.shape[0]}")
            print(f"z_dim={z_dim}, neighbor_features_dim={neighbor_features_dim}")
            return
            
        print(f"  1. Destination Z-Order Embedding (dim={z_dim}):")
        print(f"     {local_state[:z_dim]}")
        
        idx = z_dim
        agent = self.hub_agents.get(hub_id)
        if not agent:
            print(f"ERROR: Could not find agent for hub {hub_id}")
            return
            
        sorted_neighbors = agent.neighbors
        print("  2. Neighbor Features:")
        for i in range(self.max_neighbors):
            if i < len(sorted_neighbors):
                neighbor_hub_id = sorted_neighbors[i]
                print(f"     - Neighbor '{neighbor_hub_id}' (Slot {i}):")
            else:
                print(f"     - Padded Neighbor (Slot {i}):")
            
            print(f"       - Norm Travel Time to N:      {local_state[idx]:.4f}")
            idx += 1
            print(f"       - Norm Dist from N to Dest:   {local_state[idx]:.4f}")
            idx += 1
            print(f"       - N Outgoing Congestion:      {local_state[idx]:.4f}")
            idx += 1
            
        print("--- End Local State Log ---")

    def set_manual_hubs(self, hub_node_dict):
        """
        Manually set the hubs for the environment instead of using automatic selection.
        
        Args:
            hub_node_dict (dict): Dictionary mapping hub_id (e.g. 'hub_0') to node_id (SUMO junction ID)
                Example: {'hub_0': 'junction1', 'hub_1': 'junction2', ...}
                
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate the hub nodes exist in the network
        for hub_id, node_id in hub_node_dict.items():
            if node_id not in self.sumo_graph.nodes():
                print(f"Error: Node '{node_id}' for hub '{hub_id}' not found in the network.")
                return False
        
        # Set the new hubs
        self.hubs = hub_node_dict.copy()
        print(f"Successfully set {len(self.hubs)} manual hubs.")
        
        # Update hub coordinates
        self.hub_coordinates = {}
        for hub_id, node_id in self.hubs.items():
            if node_id in self.node_positions:
                self.hub_coordinates[hub_id] = self.node_positions[node_id]
            else:
                print(f"Warning: Could not find position for node '{node_id}' (hub '{hub_id}')")
        
        # Recreate hub agents
        self.hub_agents = {hub_id: HubAgent(hub_id, node_id)
                          for hub_id, node_id in self.hubs.items()}
        
        # Get ingoing edges for each hub (needed for vehicle detection)
        self._get_hub_ingoing_edges()
        
        # Calculate hub-to-hub shortest path distances for state representation
        self.hub_network_distances = {}
        for source_hub_id, source_node in self.hubs.items():
            self.hub_network_distances[source_hub_id] = {}
            for target_hub_id, target_node in self.hubs.items():
                if source_hub_id != target_hub_id:
                    # Get shortest path distance from network
                    if source_node in self.all_pairs_shortest_path and target_node in self.all_pairs_shortest_path[source_node]:
                        distance = self.all_pairs_shortest_path[source_node][target_node]
                    else:
                        # Default large distance if no path exists
                        distance = 1000.0  
                    self.hub_network_distances[source_hub_id][target_hub_id] = distance
        
        # Reset waiting vehicles count for new hubs
        self.waiting_vehicles_per_hub = {hub_id: 0 for hub_id in self.hub_agents}
        
        # Note: The hub graph and agent neighbors need to be configured after setting manual hubs
        # This happens via set_manual_hub_neighbors() or by recreating the hub graph automatically
        
        return True
    
    def set_manual_hub_neighbors(self, hub_neighbors_dict):
        """
        Manually set the neighbors for each hub instead of using automatic connections.
        Must be called after set_manual_hubs() if you're using manual hub selection.
        
        Args:
            hub_neighbors_dict (dict): Dictionary mapping hub_id to list of neighboring hub_ids
                Example: {'hub_0': ['hub_1', 'hub_2'], 'hub_1': ['hub_0', 'hub_3'], ...}
                
        Returns:
            bool: True if successful, False otherwise
        """
        # Validate all hubs and neighbors exist
        for hub_id, neighbors in hub_neighbors_dict.items():
            if hub_id not in self.hubs:
                print(f"Error: Hub '{hub_id}' not found in the environment's hubs.")
                return False
            
            for neighbor_id in neighbors:
                if neighbor_id not in self.hubs:
                    print(f"Error: Neighbor hub '{neighbor_id}' for hub '{hub_id}' not found.")
                    return False
        
        # Create a new hub graph with manual connections
        hub_graph = nx.DiGraph()
        
        # Add all hubs as nodes
        for hub_id in self.hubs.keys():
            hub_graph.add_node(hub_id, waiting_vehicles=0.0, local_congestion=0.0)
        
        # Add edges based on manual neighbor specification
        edge_count = 0
        for hub_id, neighbors in hub_neighbors_dict.items():
            for neighbor_id in neighbors:
                if hub_id == neighbor_id:
                    print(f"Warning: Skipping self-loop for hub '{hub_id}'")
                    continue
                
                # Calculate path properties between the hubs
                source_node = self.hubs[hub_id]
                target_node = self.hubs[neighbor_id]
                
                path_length = 0
                travel_time = 0
                
                # Try to find a path in the underlying network
                try:
                    path = nx.shortest_path(self.sumo_graph, source_node, target_node, weight='weight')
                    
                    # Calculate path properties
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if self.sumo_graph.has_edge(u, v):
                            edge_data = self.sumo_graph.get_edge_data(u, v)
                            if isinstance(edge_data, dict):
                                if 0 in edge_data and 'edge' in edge_data[0]:
                                    edge = edge_data[0]['edge']
                                    if hasattr(edge, 'length'):
                                        path_length += edge.length
                                    if hasattr(edge, 'length') and hasattr(edge, 'flow_speed'):
                                        travel_time += edge.length / max(0.1, edge.flow_speed)
                except nx.NetworkXNoPath:
                    # If no path exists, use default large values
                    print(f"Warning: No path found from hub '{hub_id}' to '{neighbor_id}' in the network")
                    path_length = 1000.0
                    travel_time = 1000.0
                
                # Add edge with properties
                hub_graph.add_edge(
                    hub_id, neighbor_id,
                    static_length=path_length,
                    static_travel_time=travel_time,
                    current_travel_time=travel_time,
                    current_vehicle_count=0
                )
                edge_count += 1
        
        # Set the new hub graph
        self.hub_graph = hub_graph
        print(f"Created manual hub graph with {hub_graph.number_of_nodes()} nodes and {edge_count} edges.")
        
        # Update agent neighbors
        self._populate_agent_neighbors()
        
        # Recalculate max_neighbors after manually setting connections
        if self.hub_graph.nodes():
            self.max_neighbors = max(len(list(self.hub_graph.successors(hub_id))) 
                                   for hub_id in self.hub_graph.nodes())
        else:
            self.max_neighbors = 0
        print(f"Recalculated max_neighbors after manual setup: {self.max_neighbors}")
        
        return True
    
    def rebuild_hub_graph(self):
        """
        Rebuilds the hub graph based on current hubs.
        Useful after manual hub selection if you want automatic neighbor determination.
        
        Returns:
            nx.DiGraph: The rebuilt hub graph
        """
        self.hub_graph = self._create_hub_graph()
        self._populate_agent_neighbors()
        return self.hub_graph
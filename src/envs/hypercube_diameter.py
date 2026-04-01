import numpy as np
from itertools import combinations
from collections import deque

from src.envs.environment import BaseEnvironment, DataPoint
from src.envs.tokenizers import SparseTokenizerSingleInteger
from src.utils import bool_flag


class HypercubeDiameterDataPoint(DataPoint):
    """
    Represents a spanning subgraph of d-dimensional hypercube with diameter d.
    
    The d-dimensional hypercube Q_d has:
    - 2^d vertices (each represented by a d-bit binary string)
    - d * 2^(d-1) edges (connecting vertices that differ in exactly 1 bit)
    - diameter d (maximum shortest path between any two vertices)
    
    Goal: Find a spanning subgraph with minimum number of edges while maintaining diameter d.
    
    Attributes:
        d: Dimension of the hypercube
        data: Adjacency matrix of the subgraph (2^d x 2^d)
        vertex_count: Number of vertices = 2^d
        edge_list: List of edges in the subgraph
        diameter: Current diameter of the subgraph
        score: Number of edges if valid (diameter=d), -1 otherwise
    """
    
    MAKE_OBJECT_CANONICAL = False
    
    def __init__(self, d, init=False):
        super().__init__()
        self.d = d
        self.vertex_count = 2 ** d
        self.data = np.zeros((self.vertex_count, self.vertex_count), dtype=np.uint8)
        self.edge_list = []
        self.diameter = float('inf')
        
        if init:
            self._build_graham_harary_construction()
            if self.MAKE_OBJECT_CANONICAL:
                pass  # Could add canonical form here
            self.calc_features()
            self.calc_score()
    
    def get_edges(self):
        """Return list of edges as tuples of vertex indices."""
        edges = []
        for i in range(self.vertex_count):
            for j in range(i + 1, self.vertex_count):
                if self.data[i, j] == 1:
                    edges.append((i, j))
        return edges
    
    def _vertex_to_binary(self, v):
        """Convert vertex index to d-bit binary representation."""
        return format(v, f'0{self.d}b')
    
    def _binary_to_vertex(self, binary_str):
        """Convert d-bit binary string to vertex index."""
        return int(binary_str, 2)
    
    def _get_hypercube_neighbors(self, v):
        """Get all neighbors of vertex v in the original hypercube."""
        neighbors = []
        for i in range(self.d):
            neighbor = v ^ (1 << i)  # Flip i-th bit
            neighbors.append(neighbor)
        return neighbors
    
    def _compute_shortest_path(self, start, end):
        """
        Compute shortest path between start and end using BFS.
        Returns path length or infinity if no path exists.
        """
        if start == end:
            return 0
        
        visited = {start}
        queue = deque([(start, 0)])
        
        while queue:
            current, dist = queue.popleft()
            
            # Check all neighbors in the subgraph
            for neighbor in range(self.vertex_count):
                if self.data[current, neighbor] == 1 and neighbor not in visited:
                    if neighbor == end:
                        return dist + 1
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        return float('inf')
    
    def _compute_diameter(self):
        """Compute the diameter of the current subgraph."""
        max_dist = 0
        for i in range(self.vertex_count):
            for j in range(i + 1, self.vertex_count):
                dist = self._compute_shortest_path(i, j)
                if dist == float('inf'):
                    return float('inf')  # Graph is disconnected
                max_dist = max(max_dist, dist)
        return max_dist
    
    def calc_score(self):
        """
        Calculate the score.
        Score = number of edges if diameter = d, -1 otherwise.
        """
        self.diameter = self._compute_diameter()
        
        if self.diameter != self.d:
            self.score = -1
        else:
            self.score = self.data.sum().item() // 2
    
    def calc_features(self):
        """Create a string representation for deduplication."""
        w = []
        for i in range(self.vertex_count):
            for j in range(i + 1, self.vertex_count):
                w.append(self.data[i, j])
        self.features = ",".join(map(str, w))
    
    def _build_graham_harary_construction(self):
        """
        Build the Graham-Harary construction as initial solution.
        
        For each vertex u (except v and v'), keep two edges:
        - One toward closer vertex to v = 000...0
        - One toward closer vertex to v' = 111...1
        """
        v = 0  # 000...0
        v_prime = self.vertex_count - 1  # 111...1
        
        # For each vertex, find edges toward v and v'
        for u in range(self.vertex_count):
            if u == v or u == v_prime:
                continue
            
            # Get distance to v and v' in original hypercube
            dist_to_v = bin(u).count('1')  # Hamming weight
            dist_to_v_prime = self.d - dist_to_v
            
            # Add edge toward v (flip a 1-bit to 0)
            if dist_to_v > 0:
                # Choose a random bit that is 1
                ones_bits = [i for i in range(self.d) if (u >> i) & 1]
                if ones_bits:
                    bit_to_flip = np.random.choice(ones_bits)
                    neighbor_toward_v = u ^ (1 << bit_to_flip)
                    self._add_edge(u, neighbor_toward_v)
            
            # Add edge toward v' (flip a 0-bit to 1)
            if dist_to_v_prime > 0:
                # Choose a random bit that is 0
                zero_bits = [i for i in range(self.d) if not ((u >> i) & 1)]
                if zero_bits:
                    bit_to_flip = np.random.choice(zero_bits)
                    neighbor_toward_v_prime = u ^ (1 << bit_to_flip)
                    self._add_edge(u, neighbor_toward_v_prime)
        
        # Ensure v and v' are connected
        # Add edges from v to all vertices at distance 1
        for i in range(self.d):
            neighbor = 1 << i
            self._add_edge(v, neighbor)
        
        # Add edges from v' to all vertices at distance 1 from v'
        for i in range(self.d):
            neighbor = v_prime ^ (1 << i)
            self._add_edge(v_prime, neighbor)
    
    def _add_edge(self, u, v):
        """Add an edge if it exists in the original hypercube."""
        # Check if u and v differ in exactly one bit
        diff = u ^ v
        if diff != 0 and (diff & (diff - 1)) == 0:  # Power of 2
            self.data[u, v] = 1
            self.data[v, u] = 1
    
    def _can_add_edge(self, u, v):
        """Check if edge (u,v) exists in original hypercube."""
        diff = u ^ v
        return diff != 0 and (diff & (diff - 1)) == 0
    
    def _add_edges_greedily(self):
        """
        Greedily add edges while trying to maintain diameter constraint.
        This is used for local search improvement.
        """
        # Get all possible hypercube edges not yet in our subgraph
        possible_edges = []
        for u in range(self.vertex_count):
            for v in range(u + 1, self.vertex_count):
                if self.data[u, v] == 0 and self._can_add_edge(u, v):
                    possible_edges.append((u, v))
        
        np.random.shuffle(possible_edges)
        
        # Try to add edges that don't increase diameter beyond d
        for u, v in possible_edges:
            self.data[u, v] = 1
            self.data[v, u] = 1
            
            # Quick check: does this create a shortcut that reduces diameter?
            # If yes, keep it. If it makes diameter worse, remove it.
            new_diameter = self._compute_diameter()
            if new_diameter > self.d:
                self.data[u, v] = 0
                self.data[v, u] = 0
    
    def _remove_edges_greedily(self):
        """
        Greedily remove edges to try to reduce edge count while maintaining diameter d.
        """
        edges = self.get_edges()
        np.random.shuffle(edges)
        
        for u, v in edges:
            # Temporarily remove edge
            self.data[u, v] = 0
            self.data[v, u] = 0
            
            # Check if diameter is still d
            new_diameter = self._compute_diameter()
            if new_diameter > self.d:
                # Restore edge
                self.data[u, v] = 1
                self.data[v, u] = 1
    
    def local_search(self, improve_with_local_search=True):
        """
        Apply local search to fix violations and optionally improve.
        """
        # Step 1: Compute current diameter
        self.diameter = self._compute_diameter()
        
        # Step 2: If diameter > d, we need to add edges to fix it
        if self.diameter > self.d:
            self._fix_diameter()
        
        # Step 3: Optionally try to remove edges
        if improve_with_local_search:
            self._remove_edges_greedily()
        
        # Step 4: Recompute diameter and score
        self.diameter = self._compute_diameter()
        self.calc_features()
        self.calc_score()
    
    def _fix_diameter(self):
        """
        Fix diameter by adding necessary edges.
        Find pairs with distance > d and add edges to shorten their paths.
        """
        max_iterations = 100
        iteration = 0
        
        while self.diameter > self.d and iteration < max_iterations:
            iteration += 1
            
            # Find the pair with maximum distance
            max_dist = 0
            worst_pair = None
            
            for i in range(self.vertex_count):
                for j in range(i + 1, self.vertex_count):
                    dist = self._compute_shortest_path(i, j)
                    if dist > max_dist:
                        max_dist = dist
                        worst_pair = (i, j)
            
            if worst_pair is None or max_dist <= self.d:
                break
            
            # Add edges along the shortest path in original hypercube
            u, v = worst_pair
            path = self._get_hypercube_path(u, v)
            
            for i in range(len(path) - 1):
                self._add_edge(path[i], path[i + 1])
            
            self.diameter = self._compute_diameter()
    
    def _get_hypercube_path(self, u, v):
        """Get a shortest path between u and v in the original hypercube."""
        diff = u ^ v
        path = [u]
        current = u
        
        while current != v:
            # Find a bit where current differs from v
            for i in range(self.d):
                if ((current >> i) & 1) != ((v >> i) & 1):
                    next_vertex = current ^ (1 << i)
                    path.append(next_vertex)
                    current = next_vertex
                    break
        
        return path
    
    @classmethod
    def _update_class_params(cls, pars):
        """Update class-level parameters for multiprocessing."""
        cls.MAKE_OBJECT_CANONICAL = pars
    
    @classmethod
    def _save_class_params(cls):
        """Save class-level parameters for multiprocessing."""
        return cls.MAKE_OBJECT_CANONICAL
    
    @classmethod
    def _batch_generate_and_score(cls, batch_size, N, process_pool=False, num_workers=1):
        """
        Generate random valid subgraphs.
        N here represents the dimension d.
        """
        results = []
        for _ in range(batch_size):
            dp = cls(N, init=True)
            if dp.score >= 0:
                results.append(dp)
        return results


class HypercubeDiameterEnvironment(BaseEnvironment):
    """
    Environment for the hypercube diameter problem.
    
    Problem: Find a spanning subgraph of d-dimensional hypercube
    with minimum edges while maintaining diameter d.
    """
    
    k = 2  # Edges are represented as pairs of vertices
    are_coordinates_symmetric = True  # Undirected graph
    data_class = HypercubeDiameterDataPoint
    
    def __init__(self, params):
        super().__init__(params)
        self.data_class.MAKE_OBJECT_CANONICAL = params.make_object_canonical
        
        # Calculate vertex count from dimension d
        vertex_count = 2 ** params.N
        
        if params.encoding_tokens == "single_integer":
            self.tokenizer = SparseTokenizerSingleInteger(
                self.data_class, 
                vertex_count,  # Tokenizer needs vertex count for vocabulary
                self.k, 
                self.are_coordinates_symmetric, 
                self.SPECIAL_SYMBOLS
            )
            # Mark as hypercube so tokenizer knows to convert N back to d
            self.tokenizer._is_hypercube = True
        elif params.encoding_tokens == "sequence_k_tokens":
            self.tokenizer = SparseTokenizerSequenceKTokens(
                self.data_class, 
                vertex_count,  # Tokenizer needs vertex count for vocabulary
                self.k, 
                self.are_coordinates_symmetric, 
                self.SPECIAL_SYMBOLS
            )
            self.tokenizer._is_hypercube = True
        elif params.encoding_tokens == "adjacency":
            self.tokenizer = DenseTokenizer(
                self.data_class, 
                vertex_count,  # Tokenizer needs vertex count for vocabulary
                self.k, 
                self.are_coordinates_symmetric, 
                self.SPECIAL_SYMBOLS, 
                pow2base=params.pow2base
            )
            self.tokenizer._is_hypercube = True
        else:
            raise ValueError(f"Invalid encoding: {params.encoding_tokens}")
    
    @staticmethod
    def register_args(parser):
        """Register environment-specific arguments."""
        parser.add_argument(
            "--N", 
            type=int, 
            default=5, 
            help="Dimension d of the hypercube Q_d (recommended: 5-6)"
        )
        parser.add_argument(
            "--encoding_tokens", 
            type=str, 
            default="single_integer", 
            help="Token encoding: single_integer/sequence_k_tokens/adjacency"
        )
        parser.add_argument(
            "--make_object_canonical", 
            type=bool_flag, 
            default="false", 
            help="Apply canonical form to remove symmetries"
        )
        parser.add_argument(
            "--pow2base", 
            type=int, 
            default=1, 
            help="Bits per token for adjacency encoding"
        )

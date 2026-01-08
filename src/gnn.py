"""GNN policy model with global node for action selection."""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from src.diag import DiagramState, NUM_NODE_TYPES
from src.act import Action, CNodeAction, FaceAction, CliffordAction


class GNNPolicy(nn.Module):
    """GNN-based policy for ZX diagram extraction."""
    
    def __init__(self, kind_dim: int = 8, hidden_dim: int = 64, num_layers: int = 3):
        """Initialize GNN policy.
        
        Args:
            kind_dim: Dimension of node type embeddings
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of message passing layers
        """
        super().__init__()
        
        self.kind_dim = kind_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node type embedding
        self.kind_embedding = nn.Embedding(NUM_NODE_TYPES, kind_dim)
        
        # Input dimension: kind_emb + [sin, cos, is_frontier, degree, is_boundary]
        input_dim = kind_dim + 5
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Global node projection
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Action scoring heads
        self.cnode_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # node1, node2, global
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.face_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # node, global
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.clifford_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # node, global
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def encode_graph(self, state: DiagramState) -> Tuple[torch.Tensor, torch.Tensor, Data]:
        """Encode graph state to node embeddings.
        
        Args:
            state: Diagram state
            
        Returns:
            Tuple of (node_embeddings, global_embedding, pyg_data)
        """
        vertices = list(state.graph.vertices())
        num_nodes = len(vertices)
        
        # Build node features
        node_features = []
        for v in vertices:
            features = state.get_node_features(v)
            kind_id = torch.tensor([features['kind']], dtype=torch.long)
            kind_emb = self.kind_embedding(kind_id).squeeze(0)
            
            other_feats = torch.tensor([
                features['sin_theta'],
                features['cos_theta'],
                features['is_frontier'],
                features['degree'],
                features['is_boundary'],
            ], dtype=torch.float32)
            
            node_feat = torch.cat([kind_emb, other_feats])
            node_features.append(node_feat)
        
        x = torch.stack(node_features)
        
        # Build edge index
        edges = list(state.graph.edges())
        if edges:
            edge_index = torch.tensor([[vertices.index(e[0]) for e in edges],
                                       [vertices.index(e[1]) for e in edges]], 
                                      dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # Create PyG data object
        data = Data(x=x, edge_index=edge_index)
        
        # Forward through GNN
        h = data.x
        for conv in self.convs:
            h = F.relu(conv(h, data.edge_index))
        
        # Global embedding
        global_emb = torch.mean(h, dim=0)  # Mean pooling
        global_emb = self.global_proj(global_emb)
        
        return h, global_emb, data
    
    def score_candidates(self, state: DiagramState, candidates: List[Action]) -> torch.Tensor:
        """Score candidate actions.
        
        Args:
            state: Current diagram state
            candidates: List of candidate actions
            
        Returns:
            Logits tensor of shape (num_candidates,)
        """
        if len(candidates) == 0:
            return torch.tensor([], dtype=torch.float32)
        
        # Encode graph
        node_embs, global_emb, data = self.encode_graph(state)
        
        # Map vertex IDs to indices
        vertices = list(state.graph.vertices())
        v_to_idx = {v: i for i, v in enumerate(vertices)}
        
        # Score each candidate
        logits = []
        for action in candidates:
            if isinstance(action, CNodeAction):
                u_idx = v_to_idx[action.u]
                v_idx = v_to_idx[action.v]
                u_emb = node_embs[u_idx]
                v_emb = node_embs[v_idx]
                feat = torch.cat([u_emb, v_emb, global_emb])
                logit = self.cnode_scorer(feat)
                
            elif isinstance(action, FaceAction):
                u_idx = v_to_idx[action.u]
                u_emb = node_embs[u_idx]
                feat = torch.cat([u_emb, global_emb])
                logit = self.face_scorer(feat)
                
            elif isinstance(action, CliffordAction):
                u_idx = v_to_idx[action.u]
                u_emb = node_embs[u_idx]
                feat = torch.cat([u_emb, global_emb])
                logit = self.clifford_scorer(feat)
            
            else:
                logit = torch.tensor([0.0])
            
            logits.append(logit.squeeze())
        
        return torch.stack(logits)
    
    def forward(self, state: DiagramState, candidates: List[Action]) -> torch.Tensor:
        """Forward pass: score candidates and return masked logits.
        
        Args:
            state: Diagram state
            candidates: List of candidate actions
            
        Returns:
            Logits tensor
        """
        return self.score_candidates(state, candidates)
    
    def sample_action(self, state: DiagramState, candidates: List[Action], temperature: float = 1.0) -> Tuple[Action, float, torch.Tensor]:
        """Sample action from policy.
        
        Args:
            state: Current state
            candidates: List of candidate actions
            temperature: Sampling temperature
            
        Returns:
            Tuple of (selected_action, log_prob, logits)
        """
        if len(candidates) == 0:
            raise ValueError("No candidates available")
        
        logits = self.forward(state, candidates)
        probs = F.softmax(logits / temperature, dim=0)
        
        # Sample action
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        return candidates[action_idx.item()], log_prob.item(), logits


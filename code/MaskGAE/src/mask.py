import dgl
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor
from torch_geometric.utils import to_undirected #sort_edge_index, degree
#from torch_geometric.utils.num_nodes import maybe_num_nodes

def mask_edge(G: dgl.Graph, p: float=0.7):
    edge_index = torch.Tensor([G.edges()[0].tolist(), G.edges()[1].tolist()])
    if p<0 or p>1:
        raise ValueError(f"Mask prob. has to be between 0 and 1, got {p}")
    
    e_ids = torch.arrange(edge_index.size(1), dtype=torch.long, device = edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32) #p 값으로 e_ids차원만큼의 텐서를 만들어라
    mask = torch.bernoulli(mask).to(torch.bool)
    return edge_index[:, ~mask], edge_index[:, mask]

class MaskEdge(nn.Module):
    def __init__(self, p: float=0.7, undirected: bool =True):
        super().__init__()
        self.p = p
        self.undirected = undirected
    
    def forward(self, g: dgl.Graph):
        remaining_edges, masked_edges = mask_edge(g, p=self.p)
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges
    
    def extra_repr(self):
        return f"p={self.p}, undirected={self.undirected}"
        

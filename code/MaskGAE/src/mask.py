import dgl
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor
from torch_geometric.utils import to_undirected #sort_edge_index, degree
#from torch_geometric.utils.num_nodes import maybe_num_nodes
def edge_indexing(G:dgl.DGLGraph):
    edge_index =torch.Tensor([G.edges()[0].tolist(), G.edges()[1].tolist()])
    return edge_index

def mask_edge(edge_index:torch.Tensor, p: float=0.7, undirected:bool=True):

    if p<0 or p>1:
        raise ValueError(f"Mask prob. has to be between 0 and 1, got {p}")
    
    e_ids = torch.arange(edge_index.size(1), dtype=torch.long, device = edge_index.device)
    mask = torch.full_like(e_ids, p, dtype=torch.float32) #p 값으로 e_ids차원만큼의 텐서를 만들어라
    mask = torch.bernoulli(mask).to(torch.bool)
    remaining_edges, masked_edge = edge_index[:,~mask], edge_index[:, mask]
    
    if undirected:
        remaining_edges = to_undirected(remaining_edges)
        
    return remaining_edges, masked_edge
'''
class MaskEdge(nn.Module):
    def __init__(self, p: float=0.7, undirected: bool =True):
        super().__init__()
        self.p = p
        self.undirected = undirected
    
    def forward(self, g: dgl.DGLGraph):
        edge_index = edge_indexing(g)
        remaining_edges, masked_edges = mask_edge(edge_index, p=self.p)
        if self.undirected:
            remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges
    
    def extra_repr(self):
        return f"p={self.p}, undirected={self.undirected}"
'''
        
def masking(g:dgl.DGLGraph,edge_index:torch.Tensor, p=0.7, undirected=True):
    edge_index = edge_indexing(g)
    remaining_edges, masked_edges = mask_edge(edge_index, p)
    if undirected:
        remaining_edges = to_undirected(remaining_edges)
    return remaining_edges, masked_edges
from typing import Union, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn.pytorch.conv import EGNNConv
#from torch_geometric.nn import Set2Set
from torch_geometric.utils import add_self_loops, negative_sampling,to_undirected

import sys, os
sys.path.append("/home/kathy531/Caesar-lig/code/MaskGAE")
#from src.mask import *

class Encoder(nn.Module):
    def __init__(self,
                 encoder_args: Dict[str, Union[str, int]],
                 n_input_feats: int,
                 channels: int, #hidden_dim
                 dropout_rate: float,
                 latent_embedding_size: int
                 ) -> None:
        super().__init__()
        self.latent_embedding_size = latent_embedding_size
        self.initial_node_embedding = nn.Sequential(nn.Linear(n_input_feats, channels),
                                                    nn.LayerNorm(channels),
                                                    nn.GELU(),
                                                    nn.Dropout(dropout_rate),
                                                    nn.Linear(channels, channels),
                                                    nn.LayerNorm(channels),
                                                    nn.GELU(),
                                                    nn.Dropout(dropout_rate))
        
        
        self.graph_conv = GraphConv(**encoder_args)
        self.dropout = nn.Dropout(0.3) #dropout setting
        self.activation = nn.GELU()
     
        #latent transform layers
        self.mu_transform = nn.Linear(channels*2, self.latent_embedding_size) #why channels*2?
        self.logvar_transform = nn.Linear(channels*2, self.latent_embedding_size)
    
    def reparametrize(self, mu, logvar):
           std = torch.exp(logvar)
           eps = torch.rand_like(std)
           return eps.mul(std).add_(mu)
       
    def forward(self, G: dgl.DGLGraph, edge_index: torch.Tensor):
        node_feat = G.ndata['attr']
        edge_feat = G.edata['attr']
        xyz = G.ndata['xyz']
        
        node_feat = self.initial_node_embedding( node_feat ).squeeze()
        
        node_feat = self.dropout(node_feat) #dropout setting
        node_feat = self.graph_conv(graph=G,
                                    node_feat=node_feat,
                                    edge_feat=edge_feat,
                                    coord_feat=xyz)
        #Use subgraph
        G = G.edge_subgraph(edge_index)
        #latent transform layers
        mu = self.mu_transform(node_feat)
        logvar = self.logvar_transform(node_feat)
        z = self.reparametrize(mu,logvar)
        
        return z, mu, logvar

class GraphConv(nn.Module):
    def __init__(self,
                 channels: int,
                 num_layers: int = 1,
                 edge_feat_dim: int = 7, # num of edge feat: 7
                 out_size: Optional[int] = None
                 ) -> None:
        super().__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.layer_norm = nn.LayerNorm(channels)

        self.gcn_layers = nn.ModuleList([EGNNConv(in_size=channels,
                                                 hidden_size=channels,
                                                 out_size=channels,
                                                 edge_feat_size=edge_feat_dim) for _ in range(num_layers)])
        
    def forward(self,
                graph: dgl.DGLGraph,
                node_feat: torch.Tensor,
                edge_feat: Optional[torch.Tensor] = None,
                coord_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        for i in range(self.num_layers):
            gcn_layer = self.gcn_layers[i]
            node_feat, _ = gcn_layer(graph=graph,
                                     node_feat=node_feat,
                                     edge_feat=edge_feat,
                                     coord_feat=coord_feat)
            if i != self.num_layers-1:
                node_feat = self.layer_norm(node_feat) #맨 마지막 레이어는 배치놈 수행 안함.
                
        return node_feat

class EdgeDecoder(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers=2
        ,dropout=0.5
    ):
        super().__init__()
        self.mlps = nn.ModuleList()
        
        for i in range(num_layers):
            first_channels = in_channels if i ==0 else hidden_channels
            second_chnnels = out_channels if i==num_layers-1 else hidden_channels
            self.mlps.append(nn.Linear(first_channels,second_chnnels))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
            
    def forward( self, z, G: dgl.DGLGraph, sigmoid=True, reduction=False):
        edge1 = G.edges()[0] #src edge_idx
        edge2 = G.edges()[1] #dst edge_idx
        x = z[edge1]*z[edge2]
        
        if reduction:
            x =x.mean(1)
        
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)
        
        x = self.mlps[-1](x)
        
        if sigmoid:
            return x.sigmoid()
        else:
            return x
        
'''          
class MaskGAE(nn.Module):
    def __init__(self, args, device ):
        super().__init__()
        self.device = device
        
        self.encoder = Encoder( encoder_args= args.encoder_args,
                                n_input_feats= args.n_input_feats,
                                channels = args.channels,
                                dropout_rate= args.dropout_rate)
        self.dropout = nn.Dropout(0.3)
    
    def decode(self, z):
        adj_rec = torch.sigmoid(torch.matmul(z,z.t()))
        return adj_rec

    def foward(self, G: dgl.Gragph, do_dropout):
        z = self.encoder(G, do_dropout)
        pred = self.decode(z)
        return pred,z
'''

def random_negative_sampler(edge_index, num_nodes,num_neg_samples):
    # torch.randint(low(포함), high(불포함), size(텐서사이즈))
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index) #생성된 텐서를 device와 동일한 디바이스로 이동
    return neg_edges

class MaskGAE(nn.Module):
    def __init__(self,
                 args,
                 device,
                 mask = True,
                 random_negative_sampling=True):
        super().__init__()
        self.device = device
        self.encoder = Encoder(encoder_args= args.encoder_args,
                                n_input_feats= args.n_input_feats,
                                channels = args.channels,
                                dropout_rate= args.dropout_rate,
                                latent_embedding_size = args.latent_embedding_size)
        
        self.edge_decoder = EdgeDecoder(in_channels=args.latent_embedding_size, hidden_channels=args.channels, out_channels=args.n_input_feats)
        self.mask = mask
        if random_negative_sampling:
            self.negative_sampler = random_negative_sampler
        else:
            self.negative_sampler = negative_sampling
    
    def edge_index(self, g: dgl.DGLGraph):
        return torch.Tensor([g.edges()[0].tolist(), g.edges()[1].tolist()]) 
    
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

    def forward(self, g: dgl.DGLGraph):
        edge_index = self.edge_index(g)
        
        if self.mask is not None:
            #instance는 함수와 같이 사용할 수 없다. 즉, 인스턴트를 먼저 생성한 후, forward메서드를 호출하여 마스킹 작업 수행
            remaining_edges, masked_edges = self.mask_edge(edge_index)
        
        aug_edge_index,_ = add_self_loops(edge_index)
        neg_edges = self.negative_sampler(
            aug_edge_index,
            num_nodes = g.num_nodes(),
            num_neg_smaples = masked_edges.view(2,-1).size(1),
        ).view_as(masked_edges)
        
        z, mu, logvar = self.encoder(g, remaining_edges)
        
        pos_out = self.edge_decoder(
            z, masked_edges, sigmoid=True
        )
        
        neg_out = self.edge_decoder(z, neg_edges, sigmoid =True)
        return z, mu, logvar, pos_out, neg_out
        
class EntropyModel(nn.Module):
    def __init__(
        self,
        args,
        device ,
        output_dim =1,
        
    ):
        super().__init__()
        self.device = device
        self.autoencoder = MaskGAE(args, device)
        self.entropy_module = nn.ModuleList([nn.Linear(args.latent_embedding_size, output_dim)])
        
    def forward(self, x):
        z, mu,logvar,posout, negout =self.autoencoder(x)
        
        for layer in self.entropy_module:
            z = layer(z)
        entropy =z
        
        return entropy, mu, logvar, posout, negout
        
        
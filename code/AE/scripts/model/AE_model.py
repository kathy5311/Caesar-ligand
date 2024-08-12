import torch
import torch.nn as nn
from .AE_transformer import *

class AutoEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 c = 32, #latent dimension
                 nhead = 4,
                 decoder_dim = 128,
                 latent_dim = 4,
                 output_dim = {'aromatic':1, 'numCH3':1,'ring':1,'hybrid1hot':5,'func':16, 'elem':9, 'numH':5},
                 nlayer_encoder=1,
                 nlayer_decoder=1):
        super().__init__()

        self.encoder = TransformerEncoder(input_dim = input_dim,
                                          output_dim = latent_dim,
                                          emb_dim=c, nhead=nhead,
                                          n_layer=nlayer_encoder)

        self.decoder = TransformerDecoder(input_dim = latent_dim,
                                          output_dim = decoder_dim,
                                          emb_dim=c, nhead=nhead,
                                          n_layer=nlayer_decoder)

        self.linear_elem = nn.Linear( c, output_dim['elem'] )
        self.linear_func = nn.Linear( c, output_dim['func'] )
        self.linear_nH = nn.Linear( c, output_dim['numH'])
        self.linear_aroma = nn.Linear(c, output_dim['aromatic'])
        self.linear_numCH3 = nn.Linear(c, output_dim['numCH3'])
        self.linear_ring = nn.Linear(c, output_dim['ring'])
        self.linear_hybrid = nn.Linear(c, output_dim['hybrid1hot'])
        

    def forward(self, x):

        z = self.encoder(x)
        h = self.decoder(z)

        pred = {}
        pred['elem'] = self.linear_elem(h)
        pred['func'] = self.linear_func(h)
        pred['numH'] = self.linear_nH(h)
        pred['aromatic'] = self.linear_aroma(h)
        pred['numCH3'] = self.linear_numCH3(h)
        pred['ring'] = self.linear_ring(h)
        pred['hybrid'] = self.linear_hybrid(h)
        
        #pred_border = self.linear_border(h_pair)
        return pred, z

class EntropyModel(nn.Module):
    def __init__(self,
                 input_dim,
                 c = 32, #latent dimension
                 nhead = 4,
                 decoder_dim = 128,
                 output_dim = {'aromatic':1, 'numCH3':1,'ring':1,'hybrid1hot':5,'func':16, 'elem':9, 'numH':5, 'entropy':1},
                 latent_dim = 4,
                 nlayer_encoder=1,
                 nlayer_decoder=1):
        super().__init__()

        #self.linear = nn.Linear( input_dim, 10 )
        self.autoencoder = AutoEncoder(input_dim,
                                       c=c, nhead=nhead,
                                       latent_dim = latent_dim,
                                       decoder_dim = decoder_dim,
                                       output_dim = output_dim,
                                       nlayer_encoder=nlayer_encoder,
                                       nlayer_decoder=nlayer_decoder)

        # place holder?
        self.entropy_module = nn.ModuleList([nn.Linear(latent_dim, output_dim['entropy'])])
        
    def forward(self, x):
        pred, z = self.autoencoder(x)

        for layer in self.entropy_module:
            z = layer(z)
        entropy = z

        return pred, entropy

    

        

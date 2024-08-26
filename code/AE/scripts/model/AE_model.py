import torch
import torch.nn as nn
from .AE_transformer import *

class AutoEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 c = 64, #latent dimension
                 nhead = 4,
                 decoder_dim = 128,
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
        #print('pred hybrid Dim:', pred['hybrid'].shape)
        
        #pred_border = self.linear_border(h_pair)
        return pred, z

class EntropyModel(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 c = 64, 
                 nhead = 4,
                 decoder_dim = 128,
                 output_dim = {'aromatic':1, 'numCH3':1,'ring':1,'hybrid1hot':5,'func':16, 'elem':9, 'numH':5, 'entropy':1},
                 #latent vector dimension revision(08/19), ori:4
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
        self.conf_entropy = nn.Linear(latent_dim, output_dim['entropy'])
        self.trans_entropy = nn.Linear(latent_dim, output_dim['entropy'])
        self.rot_entropy = nn.Linear(latent_dim, output_dim['entropy'])
        self.vib_entropy = nn.Linear(latent_dim, output_dim['entropy'])
        self.activation = nn.ReLU()
        
    def forward(self, x):
        pred, z = self.autoencoder(x)
        #z: (B,N_atom,latent_D)
        '''
        for layer in self.entropy_module:
            z = layer(z)'''
        

        entropy_pred={}
        entropy_pred['conf'] = self.conf_entropy(z)
        entropy_pred['vib'] = self.vib_entropy(z)
        entropy_pred['rot'] = self.rot_entropy(z)
        entropy_pred['trans'] = self.trans_entropy(z)
        #print(entropy_pred['trans'].shape)
        
        # ReLU
        entropy_pred['conf'] = self.activation(entropy_pred['conf'])
        entropy_pred['vib'] = self.activation(entropy_pred['vib'])
        entropy_pred['rot'] = self.activation(entropy_pred['rot'])
        entropy_pred['trans'] = self.activation(entropy_pred['trans'])

        
        
        #print('conf',entropy_pred['conf'])
        #print('conf shape',entropy_pred['conf'].shape)
        
        
        return pred, entropy_pred

    

        

import torch
import torch.nn as nn

class MultiHeadAttn(nn.Module):
    def __init__(self, emb_dim, n_head):
        super().__init__()
        self.W_Q, self.W_K, self.W_V = [], [], []
        self.n_head = n_head
        self.emb_dim = emb_dim
        for i in range(n_head):
            self.W_Q.append(nn.Linear(emb_dim, emb_dim // n_head, bias = False))
            self.W_K.append(nn.Linear(emb_dim, emb_dim // n_head, bias = False))
            self.W_V.append(nn.Linear(emb_dim, emb_dim // n_head, bias = False))

        self.W_Q = nn.ModuleList(self.W_Q)
        self.W_K = nn.ModuleList(self.W_K)
        self.W_V = nn.ModuleList(self.W_V)
            
        self.W_O = nn.Linear(emb_dim, emb_dim, bias = False)
        self.attention = nn.ParameterDict()

        '''
        self.feed_forward = nn.Sequential(
            nn.Linear(self.emb_dim, 4*self.emb_dim),
            nn.ReLU(),
            nn.Linear(4*self.emb_dim, self.emb_dim)
        )
        '''

    def forward(self, query, key, value, mask = None):
        # Q, K, V = (B, N, C)
        scale = torch.FloatTensor([self.emb_dim/self.n_head]).to( query.device )

        for i in range(self.n_head):
            Q, K, V = self.W_Q[i](query), self.W_K[i](key), self.W_V[i](value)

            similarity = torch.bmm(Q, K.transpose(1, 2)) / torch.sqrt(scale) 
        
            #masked attention
            if mask is not None:
                mask = mask.to(self.device)
                similarity= similarity.masked_fill_(mask==0, -float('inf'))
          
            similarity_p = nn.functional.softmax(similarity, dim = -1)
            self.attention[str(i)] = torch.bmm(similarity_p, V)

        multi_head_attn = torch.cat([self.attention[str(i)] for i in range(self.n_head)], dim = -1)

        return self.W_O(multi_head_attn)# (B, N, c)

class EncoderBlock(nn.Module):
    def __init__(self, emb_dim, nhead=4):
        super().__init__()
        self.multi_head_attn = MultiHeadAttn(emb_dim, nhead)
        self.norm1 = nn.LayerNorm(emb_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, nhead*emb_dim),
            nn.ReLU(),
            nn.Linear(nhead*emb_dim, emb_dim)
        )
        self.norm2 = nn.LayerNorm(emb_dim)
    
    def forward(self, x, mask=None):
        x2 = self.norm1(self.multi_head_attn(x, x, x, mask) + x)
        return self.norm2(self.feed_forward(x2) + x2)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, nhead, n_layer):
        super().__init__()
        self.embed = nn.Linear(input_dim, emb_dim)
        # no positional encodding necessary in our case...
        
        self.encoder_layer = nn.ModuleList(
            [EncoderBlock(emb_dim, nhead) for _ in range(n_layer)]
        )

        self.fc_out = nn.Linear(emb_dim, output_dim)

    def forward(self, x0):
        x = self.embed(x0)
        
        for encoder_block in self.encoder_layer:
            x = encoder_block(x)

        x = self.fc_out(x)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, nhead, n_layer):
        super().__init__()
        self.embed = nn.Linear(input_dim, emb_dim)

        self.decoder_layer = nn.ModuleList(
            [DecoderBlock(emb_dim, nhead) for _ in range(n_layer)]
        )

        self.fc_out = nn.Sequential(
            nn.Linear(emb_dim, output_dim),
            nn.Softmax())
        
    def forward(self, x):
        x = self.embed(x)
        
        for decoder_block in self.decoder_layer:
            x = decoder_block(x)

        out = self.fc_out(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, nhead=4):
        super().__init__()
        self.masked_multi_head_attn = MultiHeadAttn(emb_dim, nhead)
        self.norm1 = nn.LayerNorm(emb_dim)

        self.multi_head_attn = MultiHeadAttn(emb_dim, nhead)
        self.norm2 = nn.LayerNorm(emb_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, nhead*emb_dim),
            nn.ReLU(),
            nn.Linear(nhead*emb_dim, emb_dim)
        )
        self.norm3 = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):
        x2 = self.norm1(self.multi_head_attn(x, x, x, mask) + x)
        attended = self.norm2(self.multi_head_attn(x2, x2, x2, mask) + x2)
      
        return self.norm3(self.feed_forward(attended) + attended)

    

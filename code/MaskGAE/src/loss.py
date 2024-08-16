import torch
import torch.nn.functional as F
import torch.nn as nn

#0814: ce_loss에 reduction 추가
def ce_loss(pos_out, neg_out):
    sigmoid = nn.Sigmoid() #nn.Sigmoid는 인스턴스화 시킨 후 사용해야 함.
    pos_loss = F.binary_cross_entropy(sigmoid(pos_out), torch.ones_like(pos_out), reduction='mean')
    neg_loss = F.binary_cross_entropy(sigmoid(neg_out), torch.zeros_like(neg_out),reduction='mean')
    return pos_loss+neg_loss

def KL_div(mu, logvar):
    kl_loss = -0.5 * torch.sum(1 + logvar - torch.pow(mu,2) - torch.exp(logvar))
    return kl_loss
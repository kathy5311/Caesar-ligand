import torch
import numpy as np
import time
#from .features import FUNCS_TYPE, ELEMS_TYPE

class DataSet(torch.utils.data.Dataset):
    def __init__(self, targets, datanpz):
        self.targets = targets
        self.data = np.load(datanpz,allow_pickle=True)['features']

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index): #N: maximum nodes in batch
        t0 = time.time()

        data = self.data[index]
        elems = data['elems'].astype(int)
        func  = data['func'].astype(int)
        nH    = data['nH'].astype(int)

        obt = np.concatenate([elems,func,nH],axis=1)

        S = 0.0 # placeholder
        tag = self.targets[index]

        return torch.from_numpy(obt), S, tag # N x inputdim

def collate(samples):
    N = max([a.shape[0] for a,_,_ in samples])
    B = len(samples) #B
    C = samples[0][0].shape[1] #channel

    obt = torch.zeros((B,N,C))
    mask = torch.zeros((B,N))
    for i,(a,_,_) in enumerate(samples):
        obt[i,:a.shape[0],:] = a
        mask[i,:a.shape[0]] = 1.0

    Slabel = torch.tensor([S for _,S,_ in samples])
    tags = [tag for _,_,tag in samples]

    info = {'tags':tags}

    return obt, mask, Slabel, info
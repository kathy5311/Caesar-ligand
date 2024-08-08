import torch
import numpy as np
import time
#from .features import FUNCS_TYPE, ELEMS_TYPE
#이거 수정본이다.
def numH_encoding(numH_rev): #numH 1hotencoding
    numH_dict={
        0: [1,0,0,0,0],
        1: [0,1,0,0,0],
        2: [0,0,1,0,0],
        3: [0,0,0,1,0]
    }
    new_list = [numH_dict[i[1]] if i[1] in numH_dict else [0, 0, 0, 0, 1] for i in numH_rev]
    return new_list

class DataSet(torch.utils.data.Dataset):
    def __init__(self, targets, datanpz):
        #self.targets = [l[:-1] for l in open(targets,'r')]
        self.targets = targets
        #self.data = np.load(datanpz,allow_pickle=True)['features']
        self.data = datanpz

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index): #N: maximum nodes in batch
        t0 = time.time()
        #subset1|mol_2
        #data = self.data[index]
        target = self.targets[index]
        subset, mol = target.split("|")
        num=subset[6:]
        npzf = self.data+f"test_subset_new{num}_new.npz"
        data=np.load(npzf, allow_pickle=True)['features'][index]
        elems = data['elems'].astype(int)
        #energy = data['energy'].astype(float)
        #xyz = data['xyz'].astype(float)
        #ent = data['ent'].astype(float)
        numH = numH_encoding(data['numH']).astype(int)
        aromatic = data['aromatic'].astype(bool)
        numCH3 = data['numCH3'].astype(bool)
        ring = data['ring'].astype(bool)
        hybrid1hot = data['1hotHybrid'].astype(int)
        func1hot  = data['1hotFuncG'].astype(int)

        obt = np.concatenate([elems,numH,func1hot,aromatic,numCH3,ring,hybrid1hot],axis=1)
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
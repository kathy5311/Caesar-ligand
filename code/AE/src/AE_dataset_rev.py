
import torch
import numpy as np
import time
import traceback
#from .features import FUNCS_TYPE, ELEMS_TYPE
#이거 수정본이다.
def numH_encoding(numH_rev): #numH 1hotencoding
    numH_dict={
        0: [1,0,0,0,0],
        1: [0,1,0,0,0],
        2: [0,0,1,0,0],
        3: [0,0,0,1,0]
    }
    new_list = np.array([numH_dict[i[1]] if i[1] in numH_dict else [0, 0, 0, 0, 1] for i in numH_rev])
    return new_list

class DataSet(torch.utils.data.Dataset):
    def __init__(self, targets, datanpz):
        self.targets = [l[:-1] for l in open(targets,'r')]
        #print(self.targets)
        #self.data = np.load(datanpz,allow_pickle=True)['features']
        self.data = datanpz

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index): #N: maximum nodes in batch

        t0 = time.time()
        #subset1|mol_2
        #data = self.data[index]
        #print("index", index)
        target = self.targets[index]
        #print("target:", target)
        subset, mol = target.split("|")
        num=subset[6:]
        #print('num: ',num)
        npzf = self.data+f"subset_{num}_0830.npz"

        data=np.load(npzf, allow_pickle=True)
        idx = data['tags'].tolist().index(target)
        data=data['features'][idx]
        #data['tags'].tolist().index('subset15|mol_517')
        try:
            index < len(data)

            elems = torch.Tensor(data['elems'].astype(int))
            #print('elems',elems)
            #energy = data['energy'].astype(float)
            #xyz = data['xyz'].astype(float)
            #ent = data['ent'].astype(float)
            numH = data['numH'].tolist()
            numH_rev=[]
            for i in numH:
                sub=[i[0]-1,i[1]]
                numH_rev.append(sub)
    #print(numH)
            numH = torch.Tensor(numH_encoding(numH_rev).astype(int))
            #print(numH)
            #print(numH)
            aromatic = torch.Tensor(data['aromatic'].astype(bool)).unsqueeze(dim=1)
            #print(aromatic)
            numCH3 = torch.Tensor(data['numCH3'].astype(bool)).unsqueeze(dim=1)
            ring = torch.Tensor(data['ring'].astype(bool)).unsqueeze(dim=1)
            hybrid1hot = torch.Tensor(data['1hotHybrid'].astype(int))
            func1hot  = torch.Tensor(data['1hotFuncG'].astype(int))
            S = torch.Tensor(data['ent'].astype(float)) # placeholder
            obt = np.concatenate([elems,func1hot,numH,aromatic,numCH3,ring,hybrid1hot],axis=1)
            tag = self.targets[index]
            #print()
            
        except:
            return None, None, None
        return torch.from_numpy(obt), S, tag # N x inputdim

def collate(samples):
    
    try:
        samples = [s for s in samples if s[0] is not None]

        N = max([a.shape[0] for a,_,_ in samples])
        #print([a.shape[0] for a,_,_ in samples])
        #print(N)
        B = len(samples) #B
        C = samples[0][0].shape[1] #channel

        obt = torch.zeros((B,N,C))
        mask = torch.zeros((B,N))
        for i,(a,_,_) in enumerate(samples):
            obt[i,:a.shape[0],:] = a
            mask[i,:a.shape[0]] = 1.0
        '''   
        Slabel = torch.zeros((B,N_,C))
        mask_ = torch.zeros((B,N_))
        for i,(_,b,_) in enumerate(samples):
            Slabel[i,:b.shape[0],:]=b
            mask_[i,:b.shape[0]] = 1.0'''
        Slabel=[i for _,i,_ in samples]
        Slabel = torch.stack(Slabel)
        #print(Slabel)

        #print()
        tags = [tag for _,_,tag in samples]
        info = {'tags':tags}

    except Exception as e:
        print(e)
        return [],[],[],[]
    return obt, mask, Slabel ,info

import torch
import dgl
import numpy as np
import scipy
import random

#npz 파일 별 테그 담은 파일 만들기
#train 목록 파일이 dataf가 되는 것이다.
class DataSet(torch.utils.data.Dataset):
    def __init__(self, dataf, args):
        self.targets = [l[:-1] for l in open(dataf)]
        self.datapath = args.datapath
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        target = self.targets[index]
        #print(target)
        subset, mol = target.split("|") #슬래쉬 기호 붙이기
        num = subset[6:]
        
        npzf = self.datapath + '/' + f"test_subset_rev0718_{num}.npz"
        data = np.load(npzf, allow_pickle=True)
        idxs = data['tags'].tolist().index(target)
        data=data['features'][idxs]
        try:
            G, mask = make_graph(data)
        
        except:
            return None, None, None, None
        
        info = {'target': target}
        
        #ent label load
        S = torch.Tensor(data['ent'].astype(float))

        return G, mask, info, S
            


def numH_encoding(numH_rev): #numH 1hotencoding
    numH_dict={
        0: [1,0,0,0,0],
        1: [0,1,0,0,0],
        2: [0,0,1,0,0],
        3: [0,0,0,1,0]
    }
    new_list = [numH_dict[i[1]] if i[1] in numH_dict else [0, 0, 0, 0, 1] for i in numH_rev]
    return new_list

def make_graph(data):
    start_idx=torch.tensor([i[0] for i in data['bond_idx'].tolist()])
    end_idx=torch.tensor([i[1] for i in data['bond_idx'].tolist()])
    #node info
    xyz = data['xyz']
    random_xyz=random.randint(0,len(xyz))
    #print(f"{random_xyz}번째 그래프")
    xyz_res=xyz[random_xyz] # random choice for choicing a specific conformer's xyz
    mask_idx=[i for i in range(len(xyz_res))] 
    
    #node masking
    mask = torch.eye(len(xyz_res))[mask_idx].T
    
    #Graph generation
    g = dgl.graph((start_idx,end_idx))

    #Edge feature
    rot_bond=torch.tensor(data['rotBond']).float()[:,None]
    conju = torch.tensor(data['conjugated']).float()[:,None]
    bond_info = torch.tensor(data['bond_info']).float()
    edgefeats=[bond_info,rot_bond, conju]
    edgefeats = torch.cat([bond_info,rot_bond, conju],axis=1)

    #Node feature
    elems = torch.tensor(data['elems']).float()
    aromatic = torch.tensor(data['aromatic']).float()[:,None]
    numCH3 = torch.tensor(data['numCH3']).float()[:,None]
    ring = torch.tensor(data['ring']).float()[:,None]
    FuncG = torch.tensor(data['1hotFuncG']).float()
    Hybrid = torch.tensor(data['1hotHybrid']).float()
    xyz = torch.tensor(xyz_res).float()
    numH = data['numH'].tolist()
    numH_rev=[]
    for i in numH:
        sub=[i[0]-1,i[1]]
        numH_rev.append(sub)
    #print(numH)
    numH = torch.tensor(numH_encoding(numH_rev)).float()[:None]

    nodefeats=[elems, aromatic, numCH3, ring, FuncG, Hybrid, numH]
    nodefeats = np.concatenate(nodefeats, axis=1)
    nodefeats = nodefeats.astype(float)
    
    g.ndata['xyz'] = xyz
    g.ndata['attr'] = torch.from_numpy(nodefeats)
    g.edata['attr'] = edgefeats
    
    return g,mask

def collate(samples):
    Gs = []
    masks = []
    njs = []
    info = {'target':[]}
    S = []
    nfull = 0
    for g,m,i,s in samples:
        if g == None: continue
        #print('i',i)
        Gs.append(g)
        masks.append(m)
        njs.append(m.shape[1])
        nfull += m.shape[0]
        S.append(s)
        for key in info: info[key].append(i[key])

    if len(Gs) == 0:
        return [],[],[],[],[]
    
    bG = dgl.batch(Gs)

    ## should revisit in case b > 1
    mask = torch.zeros((len(Gs),nfull,max(njs))).to(bG.device)
    bi = 0
    for b,m in enumerate(masks):
        ni, nj = m.shape
        mask[b,bi:bi+ni,:nj] = m
        bi += ni
    #print("origin",mask.shape)
    #print("mask", mask.shape)
    #print("S",S)
    return bG,  mask, info, S
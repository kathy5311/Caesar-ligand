import numpy as np
import torch
import sys,os
import torch.nn.functional as F
from torch import nn
from model.AE_model import EntropyModel
import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.AE_dataset_rev import DataSet, collate
from torch.utils import data
print(print(os.path.abspath(__file__)))
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

class Arguments:
    def __init__(self):
        self.batchsize = 1
        self.maxepochs = 50
        self.LR = 1.0e-4
        self.W_REG = 0.0001
        self.trainlist = '/home/kathy531/Caesar-lig/code/notebooks/check_valid.txt'
        self.validlist = '/home/kathy531/Caesar-lig/code/notebooks/check_valid.txt'

args = Arguments()

set_params = {'datanpz':'/home/kathy531/Caesar-lig/data/new_npz0718/' #npz 경로 추가
              }#use default

model_params = {'input_dim': 38, #9 elems + 16 funcs + 5 numH + 1 aromatic + 1 numCH3 + 1 ring + 5 hybrid
                'latent_dim': 16
                }


loader_params = {
    'shuffle': False,
    'num_workers': 5 if '-debug' not in sys.argv else 1,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': args.batchsize}

def run_an_epoch(loader,model,optimizer,epoch,train=False,verbose=False):
    temp_loss = {'total':[],'recon':[],'entropy':[]}

    lossfunc = nn.MSELoss( reduction='none' )

    for i, (obt, mask, S, info) in enumerate(loader):
        if len(obt) == 0: continue
        if train: optimizer.zero_grad()

        S = S.to(device)
        obt = obt.to(device)
        mask = mask.to(device)
        pred, entropy_pred = model( obt )
        #print(torch.sum(entropy))
        tags = info['tags']
        #print("S",S)
        # make uncustomized all-category-equal weight
        per_category_weight = torch.ones(model_params['input_dim']).to(device)
        #print("category_size",per_category_weight.shape)
        #print("mask size", mask.shape)
        weight = torch.einsum('ij,k->ijk',mask,per_category_weight)
        #print('weight', weight)
        pred_concat = torch.cat([pred['elem'],pred['func'],pred['numH'],pred['aromatic'],pred['numCH3'],pred['ring'],pred['hybrid']],dim=-1) # channel dimension

        #add entropy part
        per_entropy_weight = torch.ones(1).to(device)
        weight_entropy = torch.einsum('ij,k->ijk',mask,per_entropy_weight)

        entropy_concat = torch.cat([entropy_pred['vib'],entropy_pred['rot'],entropy_pred['conf'],entropy_pred['trans']],dim=-1)
        #print('ent_concat', entropy_concat)
        entropy_weight= weight_entropy*entropy_concat
        
        entropy_weight = torch.sum(entropy_weight, dim=1)
        
        loss_recon = (weight*lossfunc( obt, pred_concat)).sum() # B x N x C ->
        loss_recon = loss_recon / mask.sum()
        #print('loss_recon', loss_recon)
        loss_S = (lossfunc(S, entropy_weight)).sum()
        loss_S = loss_S / mask.sum()

        loss = loss_recon + loss_S
        if train:
            loss.backward()
            optimizer.step()
            if verbose: 
                print(f"TRAIN Epoch {epoch} | Loss: {loss.item()} ")
                #print(f"Entropy: {entropy}")
        else:
            if verbose:
                print(f"VALID Epoch {epoch} | Loss: {loss.item()} ")
                natm = mask.sum(dim=-1).long()
               
                for b,(n,tag) in enumerate(zip(natm,tags)):
                    pred = pred_concat[b,:n,:]
                    label = obt[b,:n,:]
                    print(tag)
                    print(entropy_weight)
                    print(S)
                    #print(f"Entropy: {entropy}")
                    for i in range(n):
                        #두 개 이상의 속성에 해당 될 수 있을 경우 부등호로 표현
                        print(i,
                              'elem'+'_'+str([int(a) for a in (pred[i,:9]>0.5)])+'_'+str([int(a) for a in (label[i,:9]>0.5)])+'_'+
                              'func_'+str([int(a) for a in (pred[i,9:25]>0.5)])+'_'+str([int(a) for a in (label[i,9:25]>0.5)])+'_'+
                              'numH_'+str([int(a) for a in (pred[i,25:30]>0.5)])+'_'+str([int(a) for a in (label[i,25:30]>0.5)])+'_'+
                              'aromatic_'+str(int(pred[i,30]))+'_'+str(int(label[i,30]))+'_'+
                              'numCH3_'+str(int(pred[i,31]))+str(int(label[i,31]))+'_'+
                              'ring_'+str(int(pred[i,32]))+'_'+str(int(label[i,32]))+'_'+
                              'hybrid_'+str([int(a) for a in (pred[i,33:]>0.5)])+'_'+str(([int(a) for a in (label[i,33:]>0.5)])),     
                        )
                    print('Vib_'+str(round(entropy_weight[0][0].item(),4))+'_'+str(round(S[0][0].item(),4))+
                          '_Rot_'+str(round(entropy_weight[0][1].item(),4))+'_'+str(round(S[0][1].item(),4))+
                          '_Conf_'+str(round(entropy_weight[0][2].item(),4))+'_'+str(round(S[0][2].item(),4))+
                          '_Trans_'+str(round(entropy_weight[0][3].item(),4))+'_'+str(round(S[0][3].item(),4)))

        temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["recon"].append(loss_recon.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["entropy"].append(loss_S.cpu().detach().numpy()) #store as per-sample loss

    return temp_loss

def load_model(modelname):
    model = EntropyModel( **model_params )
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    epoch = 0

    model.to(device)
    model_state=torch.load('/home/kathy531/Caesar-lig/code/AE/scripts/models/%s/best.pkl'%modelname)
    model.load_state_dict(model_state['model_state_dict'],strict=False)
    #print("Keys in model_state['model_state_dict']:", model_state['model_state_dict'].keys())
    #print("Keys in model's state_dict:", model.state_dict().keys())
    
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.LR) #Adam + weight decay term(Not L2 regularization)


    return model, optimizer, epoch

def load_data():
    #trainlist = args.trainlist
    validlist = args.validlist

    #train_set = DataSet(trainlist, **set_params)
    #train_loader = data.DataLoader(train_set                               worker_init_fn=lambda _: np.random.seed(),**loader_params)

    valid_set = DataSet(validlist, **set_params)
    valid_loader = data.DataLoader(valid_set,
                                   worker_init_fn=lambda _: np.random.seed(),
                                   **loader_params)

    return valid_loader

def main():
    valid_loader = load_data()
    modelname = sys.argv[1]
    model, optimizer, start_epoch = load_model( modelname )

    verbose = '-v' in sys.argv

    for epoch in range(start_epoch, args.maxepochs):
        
        run_an_epoch(valid_loader, model, optimizer, epoch, train=False,verbose=verbose)




if __name__ =="__main__":
    torch.set_num_threads( int( os.getenv('SLURM_CPUS_PER_TASK', 4) ) )
    main()
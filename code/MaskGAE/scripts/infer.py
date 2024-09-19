#new_ train
import os
import sys
import torch
from model.model import EntropyModel
current_dir = os.path.dirname(os.path.abspath(__file__))
#print(current_dir)
sys.path.append(os.path.join('/home/kathy531/Caesar-lig/code/MaskGAE/'))
from src.dataset import DataSet, collate
from src.args import args_default as args
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.utils import data


args.modelname ='GAEtrain_0904'
def load_model(args_in, silent =False):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ## model
    model = EntropyModel(args_in, device)
    model.to(device)

    model_state = torch.load(f"/home/kathy531/Caesar-lig/code/MaskGAE/scripts/models/{args.modelname}/best.pkl", map_location=device)
    model.load_state_dict(model_state['model_state_dict'])
        
    return model

def load_data(dataf):
    loader_params ={
        'shuffle': True,
        'num_workers': 10,
        'pin_memory': True,
        'collate_fn': collate,
        'batch_size': 1,
        'worker_init_fn': np.random.seed()
    }
    
    #trainset = DataSet(args.dataf_train, args)
    validset = DataSet(args.dataf_valid, args)
    
    #train_loader = data.DataLoader(trainset, **loader_params)
    valid_loader = data.DataLoader(validset, **loader_params)

    return valid_loader

def run_an_epoch(model, data_loader, out_file,train=False):
    
    device = model.device
    
    nerr =0
    with open(out_file, 'w') as f:
        for i,(G, mask, info, S) in enumerate(data_loader):
            if type(G) != list:
                #entropy=entropy_pred
                entropy_pred, mu, logvar, posout, negout = model(G.to(device))
                if mu ==None: continue
                #print('entropy_pred', entropy['conf'], entropy['conf'].shape)
                S = torch.stack(S)
                #print('entropy',entropy)
                #print(entropy['trans'].shape)
                #print('mask',mask)
                #print('mask shape',mask.shape)
                mask = mask.to(device)
                S = torch.Tensor([1/20,1/10,1/40,1/10])*S
                S = S.to(device)
                #entropy masking

                per_entropy_weight = torch.ones(1).to(device)

                #print(per_entropy_weight.shape)
                weight = torch.einsum('bij,k->bik', mask, per_entropy_weight)
                entropy_concat = torch.cat([entropy_pred['vib'],entropy_pred['rot'],entropy_pred['conf'],entropy_pred['trans']],dim=-1)
                #print('entropy concat', entropy_concat.shape)
                mul_ent = weight*entropy_concat
                #print('ent[conf]',entropy_concat, entropy_concat.shape)
                #print('weight',weight,weight.shape)
                #print('weight.sum',weight.sum())
                #print('weight_1', weight_1, weight_1.shape)
                #0823/ 마스킹 수정하자 근데 어떻게 하지....?
                #print('ent[trans]',entropy_pred['trans'],entropy_pred['trans'].shape)
                #print('mul_ent', mul_ent[0], mul_ent.shape)
                mul_ent = torch.sum(mul_ent,dim=1)
                #print('mul_ent1',mul_ent,mul_ent.shape)
                #print('sum ent', mul_ent[0], mul_ent.shape)
                #print('S', S, S.shape)
                #print()
                #mask reproducing: fit on posout,negout shape
                mask_reduced = mask[:,:posout.size(0),:]
                mask_reduced = mask_reduced.to(device)
                '''
                print("posout size", posout.shape)
                print("mask size", mask.shape)
                print("mask_reduced size", mask_reduced.shape)
                print("negout size", negout.shape)
                '''

                verbl=''
                for tag,ent,S in zip(info['target'], mul_ent, S):
                    ent=ent.tolist()
                    S=S.tolist()
                    verbl +='Vib: '+str(round(ent[0],4))+' '+str(round(S[0],4))
                    verbl +=' Rot: '+str(round(ent[1],4))+' '+str(round(S[1],4))
                    verbl+=' Conf: '+str(round(ent[2],4))+' '+str(round(S[2],4))
                    verbl+=' Trans: '+str(round(ent[3],4))+' '+str(round(S[3],4))
                f.write(verbl+'\n')
                print(verbl)
            

            else: continue


def main():
    model =load_model(args)
    valid_loader = load_data(args)
    print('Start Inference')
    out_file = 'infer_0919.txt'
    with torch.no_grad():
        model.eval()
        run_an_epoch(model, valid_loader, out_file,train=False)


if __name__ == "__main__":
    torch.set_num_threads( int( os.getenv('SLURM_CPUS_PER_TASK', 4) ) )
    main()  

import sys
import os

# 현재 파일의 디렉토리 경로
current_dir = os.path.dirname(os.path.abspath(__file__))
#print(current_dir)
# src 디렉토리를 경로에 추가
sys.path.append(os.path.join(current_dir, '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.all_atom_model import MyModel
from dataset import DataSet, collate
from args import args_default as args

def load_model(args_in):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ## model
    model = MyModel(args_in, device)
    model.to(device)

    model_state= torch.load("/home/kathy531/Caesar/code/scripts/models/prac_0805_big/best.pkl", map_location=device)
    model.load_state_dict(model_state['model_state_dict'])

    return model

## load dataset
def load_data(dataf):
    loader_params = {
        'shuffle': True,
        'num_workers': 10,
        'pin_memory': True,
        'collate_fn': collate,
        'batch_size': 1,
        'worker_init_fn' : np.random.seed()
    }
    #trainset = DataSet(args.dataf_train, args)
    validset = DataSet(args.dataf_valid, args)

    #train_loader = torch.utils.data.DataLoader(trainset, **loader_params)
    valid_loader = torch.utils.data.DataLoader(validset, **loader_params)

    return valid_loader

def run_an_epoch(model, data_loader, train): #verbose delete
    import sys
    
    device = model.device

    nerr = 0


    '''for i in data_loader:
        print(i)
        #print(data_loader[i])
        print()'''

    for i,(G,label,mask,info,label_int) in enumerate(data_loader):
        if len(label) == 0:
        #    nerr += 1
        #    print('error ', nerr, i)
            continue
        if type(G) != list:
            do_dropout= train
            pred_test=model(G.to(device),do_dropout)
            if pred_test==None:
                continue
            else:
                if torch.isnan(pred_test).sum()==0:

                    pred = pred_test.squeeze() #model(G.to(device))
                elif torch.any(torch.isnan(pred_test)):

                    print("Warning: detected nan, resetting EGNN output to zero")
                    pred=torch.zeros_like(pred_test)
                    pred=pred.squeeze()
                    pred.requires_grad_(True)

                mask = mask.to(device)

                #pred와 Mask 속에 nan값 있는지 확인
                if (torch.isnan(pred).sum()!=0):
                    print("pred nan")

                if (torch.isnan(mask).sum()!=0):
                    print("mask nan")
                #print("mask size: ", mask.shape)
                #print("pred size: ", pred.shape)
                #print("mask: ", mask)
                #print("pred: ", pred)
                #print("indes_pred: ", torch.argmax(pred, dim=1))
                #mask_transpose=torch.transpose(mask,1,2)
                #print("indes_mask_trans: ", torch.argmax(mask_transpose, dim=1))
                #print("maks_transpose: ", mask_transpose.shape)
                Ssum = torch.sum(torch.einsum('bij,ik->bjk',mask,pred),dim=1)
                #print("Ssum: ", Ssum)
                #print("Ssum_size: ", Ssum.shape)
                #out을 11늘인만큼 수정함(이게 맞나,,?)
                #print("Ssum_index: ",torch.argmax(Ssum,dim=1)) #각 행마다 가장 큰 열의 인덱스 반환
                label = label.to(device)
                label_int=label_int.to(device)
                #print("label_index: ", label)

                #label =torch.tensor(label, device=device)
                #label error로 텐서로 바꿈!
                Ssum_soft=F.softmax(Ssum,dim=1)
                #range_tensor=torch.tensor([0.125,0.375,0.625,0.875,1.125,1.375,1.625,1.875,2.125,2.375,2.625])
                range_tensor=torch.tensor([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.75,1.85,1.95,2.05,2.15,2.25,2.35,2.45,2.55])
                range_tensor=range_tensor.to(device)
                ex_val=torch.sum(Ssum_soft * range_tensor, dim=1)
                ex_val=round(ex_val.item(),4)
                #ex_val=ex_val.to(device)
                
                label_int=round(label_int.item(),4)
                resname,resno=info['target'][0].split(' ')
                proname,resname=resname.split('.')
                MSE=round(abs(ex_val-label_int)**2,4)
                print('Ssum_soft',Ssum_soft)
                print('label',label)
                
                #print("tag:{0}, Ssum:{1}, label:{2}".format(info['target'], torch.argmax(Ssum_soft, dim=1).item(), label.item()))
                #print("proname, resname, resno, Ssum, label")
                #(protein_name, residue_name, residue_num, label_value, expected_value)
                print(proname, resname, resno, label_int,ex_val,label.item(),torch.argmax(Ssum_soft).item(),MSE)
                #print(proname, resname, resno, torch.argmax(Ssum_soft, dim=1).item(), label.item(),ex_val,label_int )
                #print("Ssum: ", torch.argmax(Ssum_soft, dim=1))

                """
                if verbose:
                    verbl = ''
                    for tag,S,l in zip(info['target'], torch.argmax(Ssum_soft,dim=1), label):
                        
                        verbl += f"{tag:9s} {int(S[l]):2d} {int(l):2d} | "
                        
                    print(verbl)"""

        else:
            continue


def main():
    #nan detect function
    #torch.autograd.set_detect_anomaly(True)

    model= load_model(args)
    valid_loader = load_data(args)
    
    run_an_epoch(model, valid_loader,train=False)


if __name__ == "__main__":
    main()

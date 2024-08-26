#new_ train
import os
import sys
import torch
from model.model import EntropyModel
current_dir = os.path.dirname(os.path.abspath(__file__))
#print(current_dir)
sys.path.append(os.path.join('/home/kathy531/Caesar-lig/code/MaskGAE/'))
from src.dataset import DataSet, collate
from src.loss import KL_div, ce_loss
from src.args import args_default as args
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.utils import data


args.modelname ='train_0822_t'
def load_model(args_in, silent =False):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    ## model
    model = EntropyModel(args_in, device)
    model.to(device)

    ## loss
    
    train_loss_empty = {"total":[],  "lossCE":[], "lossENT":[]}
    valid_loss_empty = {"total":[],  "lossCE":[], "lossENT":[]}
    
    epoch =0
    optimizer = torch.optim.Adam(model.parameters(), lr = args_in.LR, weight_decay=1e-5)
    
    if os.path.exists("/home/kathy531/Caesar-lig/code/MaskGAE/scripts/models/%s/model.pkl"%args_in.modelname):
        if not silent: print("Loading a checkpoint")
        checkpoint = torch.load(os.path.join("/home/kathy531/Caesar-lig/code/MaskGAE/scripts/models", args_in.modelname, "model.pkl"), map_location=device)
        
        trained_dict ={}
        model_dict = model.state_dict() #state_dict: 매개변수를 포함하는 딕셔너리객체
        model_keys = list(model_dict.keys())
        
        for key in checkpoint["model_state_dict"]:
            if key in model_keys:
                wts = checkpoint["model_state_dict"][key]
                trained_dict[key] = wts
            else:
                print("skip", key)
        
        model.load_state_dict(trained_dict, strict = False)
        
        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint['train_loss']
        valid_loss = checkpoint['valid_loss']
        if not silent: print("Restarting at epoch", epoch)
        
    else:
        if not silent: print("Training a new model")
        train_loss = train_loss_empty
        valid_loss = valid_loss_empty
        
        modelpath = os.path.join("/home/kathy531/Caesar-lig/code/MaskGAE/scripts/models", args_in.modelname)
        if not os.path.isdir(modelpath):
            if not silent: print("Creating a new dir at", modelpath)
            os.mkdir(modelpath)
            
    if not silent:
        nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Nparams:", nparams)
        
    return model, optimizer, epoch, train_loss, valid_loss

def load_data(dataf):
    loader_params ={
        'shuffle': True,
        'num_workers': 10,
        'pin_memory': True,
        'collate_fn': collate,
        'batch_size': args.nbatch,
        'worker_init_fn': np.random.seed()
    }
    
    trainset = DataSet(args.dataf_train, args)
    validset = DataSet(args.dataf_valid, args)
    
    train_loader = data.DataLoader(trainset, **loader_params)
    valid_loader = data.DataLoader(validset, **loader_params)

    return train_loader, valid_loader

def run_an_epoch(model, optimizer, data_loader, train, verbose =False):
    loss_tmp = {'total':[], 'lossCE':[], 'lossENT':[]} #lossKL removal
    device = model.device
    
    nerr =0
    
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
            Ssum_pos = torch.sum(torch.einsum('bij,ik->bjk',mask_reduced,posout),dim=1)
            Ssum_neg = torch.sum(torch.einsum('bij,ik->bjk',mask_reduced,negout),dim=1)
            #print(Ssum_pos, Ssum_neg)
            #print(mu,logvar)
            #lossKL = KL_div(mu, logvar)
            #print('lossKL:',lossKL) 
            lossCE = ce_loss(Ssum_pos,Ssum_neg)
            #print('lossCE',lossCE)
            lossMSE = nn.MSELoss()
            lossENT = lossMSE(S,mul_ent).sum()
            lossENT = lossENT/ weight.sum()
            #print('mask.sum',mask.sum())

            #print('lossENT Norm',lossENT)

            loss = lossCE+lossENT #lossKL removal
            
            if train:
                if (torch.isnan(loss).sum()==0):
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()
                else:
                    pass
                
            
            if verbose:
                verbl=''
                for tag,ent,S in zip(info['target'], mul_ent, S):
                    ent=ent.tolist()
                    S=S.tolist()
                    verbl += f"{tag:9s}\n"
                    verbl +='Vib:'+str(round(ent[0],4))+' '+str(round(S[0],4))
                    verbl +=' Rot:'+str(round(ent[1],4))+' '+str(round(S[1],4))
                    verbl+=' Conf:'+str(round(ent[2],4))+' '+str(round(S[2],4))
                    verbl+=' Trans:'+str(round(ent[3],4))+' '+str(round(S[3],4))+'\n'
                print(verbl)
            
            loss_tmp['total'].append(loss.cpu().detach().numpy())
            loss_tmp['lossCE'].append(lossCE.cpu().detach().numpy())
            #loss_tmp['lossKL'].append(lossKL.cpu().detach().numpy())
            loss_tmp['lossENT'].append(lossENT.cpu().detach().numpy())
        else: continue
    return loss_tmp

def main():
    model,optimizer,init_epoch,train_loss,valid_loss =load_model(args)
    train_loader, valid_loader = load_data(args)
    
    for epoch in range(init_epoch, args.maxepoch):
        print('epoch:', epoch)
        
        loss_t = run_an_epoch(model, optimizer, train_loader, True, verbose=True)
        
        for key in train_loss:
            train_loss[key].append(np.array(loss_t[key]))
        
        with torch.no_grad():
            model.eval()
            loss_v = run_an_epoch(model, optimizer, valid_loader, False, (args.verbose and epoch==args.maxepoch-1))
        
        for key in valid_loss:
            valid_loss[key].append(np.array(loss_v[key]))
            
        print("Train/Valid: %3d %8.4f %8.4f"%(epoch, float(np.mean(loss_t['total'])),
                                              float(np.mean(loss_v['total']))))

        print("Train/Valid CE: %3d %8.4f %8.4f"%(epoch, float(np.mean(loss_t['lossCE'])),
                                              float(np.mean(loss_v['lossCE']))))    
        #print("Train/Valid KL: %3d %8.4f %8.4f"%(epoch, float(np.mean(loss_t['lossKL'])),
                                              #float(np.mean(loss_v['lossKL']))))
        print("Train/Valid ENT: %3d %8.4f %8.4f"%(epoch, float(np.mean(loss_t['lossENT'])),
                                              float(np.mean(loss_v['lossENT']))))
                           
        if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, os.path.join("/home/kathy531/Caesar-lig/code/MaskGAE/scripts/models", args.modelname, "best.pkl"))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
        }, os.path.join("/home/kathy531/Caesar-lig/code/MaskGAE/scripts/models", args.modelname, "model.pkl"))
            

if __name__ == "__main__":
    torch.set_num_threads( int( os.getenv('SLURM_CPUS_PER_TASK', 4) ) )
    main()  

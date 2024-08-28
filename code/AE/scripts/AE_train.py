import time
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
        self.batchsize = 30
        self.maxepochs = 100
        self.LR = 1.0e-4
        self.W_REG = 0.0001
        self.trainlist = '/home/kathy531/Caesar-lig/code/notebooks/total_train_0823.txt'
        self.validlist = '/home/kathy531/Caesar-lig/code/notebooks/total_valid_0823.txt'

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

def run_an_epoch(loader,model,optimizer,epoch,train,verbose=False):
    temp_loss = {'total':[],'recon':[],'entropy':[]}
    
    data_loading_time = 0.0
    processing_time =0.0

    lossfunc = nn.MSELoss( reduction='none' )

    for i, (obt, mask, S, info) in enumerate(loader):
        print(i, "starting epoch!")
        start_time = time.time()
        if len(obt) == 0: continue
        if train: optimizer.zero_grad()
        
        data_loading_time += time.time()-start_time
        start_time = time.time()

        S = S.to(device) #내가 추가한거
        obt = obt.to(device)
        #print('obt', obt)
        mask = mask.to(device)

        pred, entropy_pred = model( obt ) #origin: pred,_=model(obt)
        #print('entropy_pred',entropy_pred)
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
        print('entropy_weight.shape', entropy_weight.shape)
        loss_S = (lossfunc(S, entropy_weight)).sum()
        loss_S = loss_S / mask.sum()

        #loss_S = torch.tensor(0.0).to(device)

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
                    #print(f"Entropy: {entropy}")
                    for i in range(n):
                        #두 개 이상의 속성에 해당 될 수 있을 경우 부등호로 표현
                        print(i,
                              'elem:',[int(a) for a in (pred[i,:9]>0.5)],[int(a) for a in (label[i,:9]>0.5)],
                              'func:',[int(a) for a in (pred[i,9:25]>0.5)], [int(a) for a in (label[i,9:25]>0.5)],
                              'numH:',int(torch.argmax(pred[i,25:30])),int(torch.argmax(label[i,25:30])),
                              'aromatic',int(pred[i,30]), int(label[i,30]),
                              'numCH3', int(pred[i,31]), int(label[i,31]),
                              'ring', int(pred[i,32]), int(label[i,32]),
                              'hybrid', int(torch.argmax(pred[i,32:])),int(torch.argmax(label[i,32:]))
                        )
#0809 여기부터 수정 시작해라->check
        processing_time += time.time() - start_time
        
        temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["recon"].append(loss_recon.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["entropy"].append(loss_S.cpu().detach().numpy()) #store as per-sample loss
    print(f"Data loading time: {data_loading_time:.4f}")
    print(f"Processing time: {processing_time: .4f}")
    return temp_loss

def load_model(modelname):
    model = EntropyModel( **model_params )
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    train_loss = {'total':[],'recon':[],'entropy':[]}
    valid_loss = {'total':[],'recon':[],'entropy':[]}
    epoch = 0

    model.to(device)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.LR) #Adam + weight decay term(Not L2 regularization)

    if os.path.exists("/home/kathy531/Caesar-lig/code/AE/scripts/models/%s/model.pkl"%modelname):
        checkpoint = torch.load("/home/kathy531/Caesar-lig/code/AE/scripts/models/"+modelname+"/model.pkl", map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        print("Restarting at epoch", epoch)
    else:
        if not os.path.exists("/home/kathy531/Caesar-lig/code/AE/scripts/models/%s"%(modelname)):
            print("Creating a new dir at models/%s"%modelname)
            os.mkdir("/home/kathy531/Caesar-lig/code/AE/scripts/models/"+modelname)

    return model, optimizer, epoch, train_loss, valid_loss

def load_data():
    trainlist = args.trainlist
    validlist = args.validlist

    train_set = DataSet(trainlist, **set_params)
    train_loader = data.DataLoader(train_set,
                                   worker_init_fn=lambda _: np.random.seed(),
                                   **loader_params)

    valid_set = DataSet(validlist, **set_params)
    valid_loader = data.DataLoader(valid_set,
                                   worker_init_fn=lambda _: np.random.seed(),
                                   **loader_params)

    return train_loader, valid_loader

def main():
    train_loader, valid_loader = load_data()
    modelname = sys.argv[1]
    model, optimizer, start_epoch, train_loss, valid_loss = load_model( modelname )

    verbose = '-v' in sys.argv

    for epoch in range(start_epoch, args.maxepochs):
        
        temp_loss = run_an_epoch(train_loader, model, optimizer, epoch, True)
        for key in temp_loss:
            train_loss[key].append(temp_loss[key])

        with torch.no_grad():
            temp_loss = run_an_epoch(valid_loader, model, optimizer, epoch, False, verbose=verbose)
            for key in temp_loss:
                valid_loss[key].append(temp_loss[key])

        print("Epoch %d, train/valid loss: %7.4f %7.4f"%((epoch,
                                                          np.mean(train_loss['total'][-1]),
                                                          np.mean(valid_loss['total'][-1]))))
        print("Epoch %d, train/valid Reconloss: %7.4f %7.4f"%((epoch,
                                                          np.mean(train_loss['recon'][-1]),
                                                          np.mean(valid_loss['recon'][-1]))))
        print("Epoch %d, train/valid Entropyloss: %7.4f %7.4f"%((epoch,
                                                          np.mean(train_loss['entropy'][-1]),
                                                          np.mean(valid_loss['entropy'][-1]))))


        # Update the best model if necessary:
        if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss}, "/home/kathy531/Caesar-lig/code/AE/scripts/models/"+modelname+"/best.pkl")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss}, "/home/kathy531/Caesar-lig/code/AE/scripts/models/"+modelname+"/model.pkl")


if __name__ =="__main__":
    torch.set_num_threads( int( os.getenv('SLURM_CPUS_PER_TASK', 4) ) )
    main()
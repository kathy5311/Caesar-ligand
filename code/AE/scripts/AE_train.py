import numpy as np
import torch
import sys,os
import torch.nn.functional as F
from torch import nn
from .model_0711 import EntropyModel
from .dataset_0711 import DataSet, collate
from torch.utils import data

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

class Arguments:
    def __init__(self):
        self.batchsize = 5
        self.maxepochs = 100
        self.LR = 1.0e-4
        self.W_REG = 0.0001
        self.trainlist = 'data/trainlist.npy'
        self.validlist = 'data/validlist.npy'

args = Arguments()

set_params = {'datanpz':'data/test.npz'
              }#use default

model_params = {'input_dim': 17, #9 elems + 4 funcs + 4 nH
                'latent_dim': 2
                }


loader_params = {
    'shuffle': False,
    'num_workers': 5 if '-debug' not in sys.argv else 1,
    'pin_memory': True,
    'collate_fn': collate,
    'batch_size': args.batchsize}

def run_an_epoch(loader,model,optimizer,epoch,train,verbose=False):
    temp_loss = {'total':[],'recon':[],'entropy':[]}

    lossfunc = nn.MSELoss( reduction='none' )

    for i, (obt, mask, S, info) in enumerate(loader):
        if train: optimizer.zero_grad()

        obt = obt.to(device)
        mask = mask.to(device)
        pred, entropy = model( obt )
        tags = info['tags']

        # make uncustomized all-category-equal weight
        per_category_weight = torch.ones(model_params['input_dim']).to(device)
        weight = torch.einsum('ij,k->ijk',mask,per_category_weight)

        pred_concat = torch.cat([pred['elem'],pred['func'],pred['nH']],dim=-1) # channel dimension

        loss_recon = (weight*lossfunc( obt, pred_concat )).sum() # B x N x C ->
        loss_recon = loss_recon / mask.sum()

        loss_S = torch.tensor(0.0).to(device)

        loss = loss_recon + loss_S
        if train:
            loss.backward()
            optimizer.step()
            if verbose: print(f"TRAIN Epoch {epoch} | Loss: {loss.item()} ")
        else:
            if verbose:
                print(f"VALID Epoch {epoch} | Loss: {loss.item()} ")
                natm = mask.sum(dim=-1).long()
                for b,(n,tag) in enumerate(zip(natm,tags)):
                    pred = pred_concat[b,:n,:]
                    label = obt[b,:n,:]
                    print(tag)
                    for i in range(n):
                        print(i,
                              'elem:',int(torch.argmax(pred[i,:9])),int(torch.argmax(label[i,:9])),
                              'func:',[int(a) for a in (pred[i,9:13]>0.5)], [int(a) for a in (label[i,9:13]>0.5)],
                              'nH:',int(torch.argmax(pred[i,13:])),int(torch.argmax(label[i,13:]))
                        )

        temp_loss["total"].append(loss.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["recon"].append(loss_recon.cpu().detach().numpy()) #store as per-sample loss
        temp_loss["entropy"].append(loss_S.cpu().detach().numpy()) #store as per-sample loss

    return temp_loss

def load_model(modelname):
    model = EntropyModel( **model_params )
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    train_loss = {'total':[],'recon':[],'entropy':[]}
    valid_loss = {'total':[],'recon':[],'entropy':[]}
    epoch = 0

    model.to(device)
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.LR)

    if os.path.exists("models/%s/model.pkl"%modelname):
        checkpoint = torch.load("models/"+modelname+"/model.pkl", map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"],strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epoch = checkpoint["epoch"]+1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        print("Restarting at epoch", epoch)
    else:
        if not os.path.exists("models/%s"%(modelname)):
            print("Creating a new dir at models/%s"%modelname)
            os.mkdir("models/"+modelname)

    return model, optimizer, epoch, train_loss, valid_loss

def load_data():
    trainlist = np.load(args.trainlist)
    validlist = np.load(args.validlist)

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

        # Update the best model if necessary:
        if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss}, "models/"+modelname+"/best.pkl")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss}, "models/"+modelname+"/model.pkl")


if __name__ =="__main__":
    main()
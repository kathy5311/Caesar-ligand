class Argument:
    def __init__(self, modelname, dropout_rate=0.3):
        self.modelname = modelname
        self.dropout_rate = dropout_rate
        self.nbatch =10
        
        self.maxepoch = 50
        self.datapath = '/home/kathy531/Caesar-lig/data/new_npz0718'
        self.dataf_train = '/home/kathy531/Caesar-lig/code/notebooks/check_train.txt'
        self.dataf_valid = '/home/kathy531/Caesar-lig/code/notebooks/check_valid.txt'
        self.LR = 1.0e-5 #1.0e-4
        self.topk = 32
        self.n_input_feats = 38
        self.channels = 64
        self.latent_embedding_size =2

        self.encoder_args = {'channels': 64, 'num_layers': 8}
        self.decoder_args = {'channels': 64, 'num_layers': 8, 'edge_feat_dim': 7}

        self.verbose = True
        
args_default = Argument("default")
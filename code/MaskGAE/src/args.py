class Argument:
    def __init__(self, modelname, dropout_rate=0.3):
        self.modelname = modelname
        self.dropout_rate = dropout_rate
        self.nbatch =30
        
        self.maxepoch = 100
        self.datapath = '/home/kathy531/Caesar-lig/data/new_npz0718'
        self.dataf_train = '/home/kathy531/Caesar-lig/code/notebooks/total_train_0823.txt'
        self.dataf_valid = '/home/kathy531/Caesar-lig/code/notebooks/total_valid_0823.txt'
        self.LR = 1.0e-5 #1.0e-4
        self.topk = 32
        self.n_input_feats = 38
        self.channels = 64
        self.latent_embedding_size =16

        self.encoder_args = {'channels': 64, 'num_layers': 8}
        self.decoder_args = {'channels': 64, 'num_layers': 8, 'edge_feat_dim': 7}

        self.verbose = True
        
args_default = Argument("default")
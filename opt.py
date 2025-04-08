import os
import datetime
import argparse
import random
import numpy as np
import torch 
import logging.config

class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of MSN-GGN')
        parser.add_argument('--n_folds', type=int, default=5, help='number of k-fold cross validation') 
        parser.add_argument('--node', type=str, default='Wei-CC', help='kBest, PCA, RFE, Wei - Strength, Betweenness, CC, Absstrength, Eigen, Concat')
        parser.add_argument('--quantile', type=float, default=0.6, help='threshold quantile')
        parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
        parser.add_argument('--wd', default=5e-5, type=float, help='weight decay')
        parser.add_argument('--num_iter', default=300, type=int, help='number of epochs for training')
        parser.add_argument('--encoder', type=str, default='PAE', help='PAE, EA, Tanh, and L2')
        parser.add_argument('--convolution', type=str, default='ChebConv', help='ChebConv, GCNConv')
        parser.add_argument('--ehidden', type=int, default=64, help='encoder hidden layer')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')
        parser.add_argument('--hgc',  nargs='+', type=int, default=[128, 64, 32], help='hidden units of gconv layer')
        parser.add_argument('--dropout', default=0.2, type=float, help='ratio of dropout')
        parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
        parser.add_argument('--node_ftr_dim', type=int, default=2000, help='number of node features (when you use RFE, K-best)')
        parser.add_argument('--K', type=int, default=3, help='chebyshev filter size K')
        parser.add_argument('--msn_threshold', type=float, default=1.0, help='setting threshold for MSN')
        parser.add_argument('--static', nargs='+', type=str, default=['SEX', 'SITE_ID'], help='nonimage value to be used when calculating initial edge')
        parser.add_argument('--ckpt_path', type=str, default='./save_models/gcn', help='checkpoint path to save trained models')
        parser.add_argument('--scheduler', type=str, default='cosine', help='None,  cosine')
        parser.add_argument('--use_encoding', type=str, default="label", help='label, embed, binary, onehot, softlabel')
        parser.add_argument('--e_dropout_rate', default=0.0, type=float, help='ratio of edge dropout')
        
        parser.add_argument('--exepoch', type=int, default=200)
        parser.add_argument('--exlr', type=float, default=0.001)
        parser.add_argument('--exmask', type=str, default="attributes")

        args = parser.parse_args()
        args.time = datetime.datetime.now().strftime("%y%m%d")

        if args.use_cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            print(" Using GPU in torch")

        self.args = args

    def print_args(self):
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")

    def initialize(self):
        self.set_seed(123)
        #self.logging_init()
        self.print_args()
        return self.args

    def set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        
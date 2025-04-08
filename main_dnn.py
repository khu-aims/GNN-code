import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from opt import * 
from DNN import DNN
from utils.metrics import balanced_accuracy, auc, prf, confusion
from dataloader import dataloader
from datetime import datetime
from time import time
from tqdm import tqdm
import pandas as pd
import copy
from collections import defaultdict
#from torch_geometric.explain.algorithm import GNNExplainer
import time

now = datetime.now()
dt_string = now.strftime("%m_%d_%H_%M")
train_time = []

if __name__ == '__main__':
    opt = OptInit().initialize()

    print('  Loading dataset ...')
    dl = dataloader() 
    raw_features, y, nonimg, site = dl.load_data(opt)
    n_folds = opt.n_folds
    cv_splits = dl.data_split(n_folds)

    accs = np.zeros(n_folds, dtype=np.float32) 
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds,3], dtype=np.float32)
    confus = np.zeros([n_folds,2], dtype=np.float32)
    
    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold)) 

        train_ind = cv_splits[fold][0] 
        test_ind = cv_splits[fold][1] 
        
        print('  Constructing graph data...')
        # extract node features  
        node_ftr = dl.get_node_features(train_ind, opt.node, opt.node_ftr_dim)

        # build network architecture  
        model = DNN(input_dim=node_ftr.shape[1], hidden_dims=opt.hgc, output_dim=opt.num_classes, 
                    dropout_rate=opt.dropout).to(opt.device)
        
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        
        if opt.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_iter, eta_min=0.00001)

        features_cuda = torch.tensor(node_ftr, dtype=torch.float32).to(opt.device)
        labels = torch.tensor(y, dtype=torch.long).to(opt.device)
        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)

        best_model_state = None 

        def train(): 
            global best_model_state, train_time
            start_time = time.time()

            print("  Number of training samples %d" % len(train_ind))
            print("  Start training...\r\n")
            max_acc = 0
            min_loss = 100.0

            for epoch in range(opt.num_iter):
                model.train()  
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(True):
                    node_logits = model(features_cuda)
                    loss = loss_fn(node_logits[train_ind], labels[train_ind])
                    loss.backward()
                    optimizer.step()

                    if opt.scheduler is not None:
                        scheduler.step()

                acc_train = balanced_accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])  
                
                model.eval()
                with torch.set_grad_enabled(False):
                    node_logits = model(features_cuda)

                logits_test = node_logits[test_ind].detach().cpu().numpy()
                acc_test = balanced_accuracy(logits_test, y[test_ind])
                loss_test = loss_fn(node_logits[test_ind], labels[test_ind])
                
                print("Epoch: {}, ce loss: {:.5f}, train acc: {:.5f}, val acc: {:.5f}, val loss: {:.5f}".format(epoch, loss.item(), acc_train.item(),  acc_test.item(), loss_test.item()))

                if min_loss > loss_test.item() and epoch > 9:
                    min_loss = loss_test.item()
                    max_acc = acc_test
                    best_model_state = copy.deepcopy(model.state_dict())

            end_time = time.time()
            train_time.append(end_time - start_time) 
            print(train_time)
            print("\r\n => Fold {} validation accuracy {:.5f}".format(fold, max_acc))

        def evaluate():
            print("  Number of testing samples %d" % len(test_ind))
            print('  Start testing...')
            
            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            model.eval()
            node_logits = model(features_cuda)
            logits_test = node_logits[test_ind].detach().cpu().numpy()

            correct_predictions = np.equal(np.argmax(logits_test, 1), y[test_ind])
            correct_counts = defaultdict(int)
            incorrect_counts = defaultdict(int)

            for is_correct, s in zip(correct_predictions, site[test_ind]):
                if is_correct: correct_counts[s] += 1
                else: incorrect_counts[s] += 1

            for s in sorted(correct_counts):
                percentage = correct_counts[s] / (incorrect_counts[s] + correct_counts[s]) * 100  # 백분율로 변환
                print(f"Site {s}: O = {correct_counts[s]}, X = {incorrect_counts[s]}, % = {percentage:.2f}%")

            accs[fold] = balanced_accuracy(logits_test, y[test_ind])
            aucs[fold] = auc(logits_test,y[test_ind]) 
            prfs[fold]  = prf(logits_test,y[test_ind])  
            confus[fold]  = confusion(logits_test,y[test_ind])  
              
            print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))

        train()
        evaluate()
   
    print("\r\n========================== Finish ==========================") 
    n_samples = raw_features.shape[0]
    acc_nfold = np.mean(accs)
    acc_nfold_std = np.std(accs)
    auc_nfold = np.mean(aucs)
    auc_nfold_std = np.std(aucs)
    print("=> Average test accuracy in {}-fold CV: {:.5f}".format(n_folds, acc_nfold))
    print("=> Average test AUC in {}-fold CV: {:.5f}".format(n_folds, auc_nfold))
    pre, rec, f1 = np.mean(prfs,axis=0)
    pre_std, rec_std, f1_std = np.std(prfs,axis=0)
    sen, spe = np.mean(confus,axis=0)
    sen_std, spe_std = np.std(confus,axis=0)

    print("=> Average test sensitivity {:.4f}, specificity {:.4f}, precision {:.4f}, recall {:.4f}, F1-score {:.4f}".format(sen, spe, pre, rec, f1))

    arg_dict = vars(opt)
    arg_dict["hgc"] = [arg_dict["hgc"]]
    arg_dict["static"] = [arg_dict["static"]]
    arg_dict["bac_mean"] = acc_nfold
    arg_dict["sen_mean"] = sen
    arg_dict["spe_mean"] = spe
    arg_dict["pre_mean"] = pre
    arg_dict["rec_mean"] = rec
    arg_dict["f1_mean"] = f1
    arg_dict["auc_mean"] = auc_nfold

    arg_dict["bac_std"] = acc_nfold_std
    arg_dict["sen_std"] = sen_std
    arg_dict["spe_std"] = spe_std
    arg_dict["pre_std"] = pre_std
    arg_dict["rec_std"] = rec_std
    arg_dict["f1_std"] = f1_std
    arg_dict["auc_std"] = auc_nfold_std
    arg_dict["train_time"] = [train_time]

    df = pd.DataFrame(arg_dict)

    if not os.path.exists('./results+DNN.csv'):
        df.to_csv('./results+DNN.csv', index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv('./results+DNN.csv', index=False, mode='a', encoding='utf-8-sig', header=False)

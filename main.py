import os
import torch
import numpy as np
from opt import * 
from GCN import GCN
from utils.metrics import balanced_accuracy, auc, prf, confusion
from dataloader import dataloader
from datetime import datetime
import pandas as pd
import copy
from torch_geometric.explain import Explainer, GNNExplainer, GraphMaskExplainer
import gc
from tqdm import tqdm
import time

now = datetime.now()
dt_string = now.strftime("%m_%d_%H_%M")
train_time = []
all_accs_train = []
all_loss_train = []
all_accs_test = []
all_loss_test = []

if __name__ == '__main__':
    opt = OptInit().initialize()
    n_folds = opt.n_folds

    print('  Loading dataset ...')
    dl = dataloader() 

    raw_features, y, nonimg, site = dl.load_data(opt)
    cv_splits = dl.data_split(n_folds)

    accs = np.zeros(n_folds, dtype=np.float32) 
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds,3], dtype=np.float32)
    confus = np.zeros([n_folds,2], dtype=np.float32)
    node_feat_masks = np.array([])
    
    for fold in range(n_folds):
        sub_accs_train = []
        sub_loss_train = []
        sub_accs_test = []
        sub_loss_test = []

        print("\r\n========================== Fold {} ==========================".format(fold)) 
        train_ind = cv_splits[fold][0] 
        test_ind = cv_splits[fold][1] 
        
        print('  Constructing graph data...')
        # extract node features  
        node_ftr = dl.get_node_features(train_ind, opt.node, opt.node_ftr_dim)

        # get inputs of encoder
        edge_index, edgenet_input, thres = dl.get_ENCODER_inputs(nonimg, opt.static, opt.quantile) 
        
        # normalization for encoder
        if opt.use_encoding == "label":
            edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)
            edgenet_input = np.nan_to_num(edgenet_input)
            print(edgenet_input)

        # build network architecture  
        model = GCN(node_ftr.shape[1], opt.num_classes, opt.dropout, hgc=opt.hgc, edgenet_input_dim= 2 * nonimg.shape[1], encoder = opt.encoder, 
                        ehidden = opt.ehidden, conv_type=opt.convolution, K=opt.K, e_dropout_rate = opt.e_dropout_rate).to(opt.device)
        
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        
        if opt.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_iter, eta_min=0.0001)

        features_cuda = torch.tensor(node_ftr, dtype=torch.float32).to(opt.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(opt.device)
        edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(opt.device)
        labels = torch.tensor(y, dtype=torch.long).to(opt.device)
        fold_model_path = opt.ckpt_path + "/ours_fold{}.pth".format(fold)
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
                    node_logits = model(features_cuda, edge_index, edgenet_input)
                    loss = loss_fn(node_logits[train_ind], labels[train_ind])
                    loss.backward()
                    optimizer.step()

                    if opt.scheduler is not None:
                        scheduler.step()

                acc_train = balanced_accuracy(node_logits[train_ind].detach().cpu().numpy(), y[train_ind])  
                sub_accs_train.append(acc_train)
                sub_loss_train.append(loss.item())

                model.eval()
            
                with torch.set_grad_enabled(False):
                    node_logits = model(features_cuda, edge_index, edgenet_input)

                logits_test = node_logits[test_ind].detach().cpu().numpy()
                acc_test = balanced_accuracy(logits_test, y[test_ind])
                loss_test = loss_fn(node_logits[test_ind], labels[test_ind])
                sub_accs_test.append(acc_test)
                sub_loss_test.append(loss_test.item())

                print("Epoch: {}, ce loss: {:.5f}, train acc: {:.5f}, val acc: {:.5f}, val loss: {:.5f}".format(epoch, loss.item(), acc_train.item(),  acc_test.item(), loss_test.item()))

                if min_loss > loss_test.item() and epoch > 9:
                    min_loss = loss_test.item()
                    max_acc = acc_test
                    best_model_state = copy.deepcopy(model.state_dict())
                   
                    #if opt.ckpt_path !='':
                    #    if not os.path.exists(opt.ckpt_path): 
                    #        os.makedirs(opt.ckpt_path)
                    #    torch.save(model.state_dict(), fold_model_path)

            end_time = time.time()
            train_time.append(end_time - start_time) 
            print("\r\n => Fold {} validation accuracy {:.5f}".format(fold, max_acc))
            all_accs_train.append(sub_accs_train)
            all_loss_train.append(sub_loss_train)
            all_accs_test.append(sub_accs_test)
            all_loss_test.append(sub_loss_test)

        def evaluate():
            print("  Number of testing samples %d" % len(test_ind))
            print('  Start testing...')
            
            #model.load_state_dict(torch.load(fold_model_path))

            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            model.eval()
            node_logits = model(features_cuda, edge_index, edgenet_input)
            logits_test = node_logits[test_ind].detach().cpu().numpy()

            accs[fold] = balanced_accuracy(logits_test, y[test_ind])
            aucs[fold] = auc(logits_test,y[test_ind]) 
            prfs[fold]  = prf(logits_test,y[test_ind])  
            confus[fold]  = confusion(logits_test,y[test_ind])  
              
            print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))

        def explain():
            global node_feat_masks
            print("Explaining . . .")

            model.load_state_dict(torch.load(fold_model_path))

            #if best_model_state is not None:
            #    model.load_state_dict(best_model_state)

            model.eval()
            
            explainer = Explainer(
                model=model,
                algorithm=GNNExplainer(epochs=opt.exepoch, lr = opt.exlr),
                explanation_type='model',
                node_mask_type=opt.exmask,
                edge_mask_type=None,
                model_config=dict(
                    mode='multiclass_classification',
                    task_level='node',
                    return_type='raw',
                ),
            )

            explanation = explainer(x = features_cuda, edge_index = edge_index, index = test_ind, edgenet_input = edgenet_input)
            if opt.exmask == "attributes":
                explanation_numpy = explanation.node_mask[test_ind].detach().cpu().numpy()
            else:
                explanation_numpy = explanation.node_mask.detach().cpu().numpy()

            if fold == 0 :
                node_feat_masks = explanation_numpy
            else:
                node_feat_masks = np.concatenate((node_feat_masks, explanation_numpy), axis=0)
            
            
        train()
        evaluate()
        # explainable gnn 사용 시 주석 해제    
        #explain()
   
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
    arg_dict["all_accs_train"] = [all_accs_train]
    arg_dict["all_loss_train"] = [all_loss_train]
    arg_dict["all_accs_test"] = [all_accs_test]
    arg_dict["all_loss_test"] = [all_loss_test]
    
    df = pd.DataFrame(arg_dict)

    if not os.path.exists('./results.csv'):
        df.to_csv('./results.csv', index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv('./results.csv', index=False, mode='a', encoding='utf-8-sig', header=False)


    '''
    # explainable gnn 사용 시 주석 해제    
    node_feat_masks = []   
    idx = []

    for fold in range(n_folds):       
        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)
        
        print(" Start expalaining...")
        model.load_state_dict(torch.load(fold_model_path))
        model.eval()

        x = features_cuda
        explainer = GNNExplainer(model, epochs=100, allow_edge_mask=False, log=False, 
                                 feat_mask_type="feature", num_hops=3)
        

        node_indices = cv_splits[fold][1] 
        
        for node_idx in tqdm(node_indices, desc='explain node', leave=False):
            idx.append(node_idx.item())

            node_feat_mask, _ = explainer.explain_node(node_idx.item(), x, edge_index,edgenet_input = edgenet_input)
            node_feat_mask = node_feat_mask.detach().cpu().numpy()
            node_feat_masks.append(node_feat_mask)
        
    df = pd.DataFrame(node_feat_masks)
    df.index = idx
    df.to_csv(f"./GNNExplainer/fold{fold}_{opt.node}_{opt.msn_threshold}_{dt_string}_node_feat.csv")
    '''
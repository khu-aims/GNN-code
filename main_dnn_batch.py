import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.metrics import balanced_accuracy, prf, auc, confusion
from dataloader import dataloader
import pandas as pd
from datetime import datetime
from opt import * 
import time
from sklearn.model_selection import StratifiedKFold

import torch.nn.functional as F
from torch.nn import Linear as Lin, Sequential as Seq

class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.5):
        super(DNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout_rate
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_batchnorm = nn.BatchNorm1d(hidden_dims[0])

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_batchnorms = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.hidden_batchnorms.append(nn.BatchNorm1d(hidden_dims[i+1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, x):
        # Input layer
        x = F.relu(self.input_batchnorm(self.input_layer(x)))
        x = F.dropout(x, self.dropout, self.training)
        
        # Hidden layers
        for layer, batchnorm in zip(self.hidden_layers, self.hidden_batchnorms):
            x = F.relu(batchnorm(layer(x)))
            x = F.dropout(x, self.dropout, self.training)
        
        # Output layer
        x = self.output_layer(x)
        return x


train_time = []

if __name__ == '__main__':
    opt = OptInit().initialize()
    print('Loading dataset ...')
    dl = dataloader()
    raw_features, y, nonimg, site = dl.load_data(opt)

    raw_features = torch.tensor(raw_features, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    batch_size = 256
    epochs = 300

    accs, aucs, prfs, confus = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(raw_features, y)):
        print(f'Fold {fold+1}')
        
        X_train, X_test = raw_features[train_idx], raw_features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        input_dim = X_train.shape[1]
        output_dim = len(torch.unique(y_train))
        model = DNN(input_dim=input_dim, hidden_dims=opt.hgc, output_dim=opt.num_classes, 
                    dropout_rate=opt.dropout).to(opt.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=opt.wd)
        
        best_loss = float('inf')
        best_model_state = None

        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X.to(opt.device))
                loss = criterion(outputs, batch_y.to(opt.device))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
            
            # 최저 loss일 때 모델 저장
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict()

        end_time = time.time()
        train_time.append(end_time - start_time) 

        # 최적 모델 로드
        model.load_state_dict(best_model_state)
        model.eval()

        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                outputs = model(inputs.to(opt.device))
                probs = outputs.softmax(dim=1)
                y_true.extend(labels.tolist())
                y_pred.extend(probs.tolist())
                y_prob.extend(probs[:, 1].tolist())  # Assuming binary classification
        
        # Fold별 성능 측정
        bal_acc = balanced_accuracy(y_pred, y_true)
        auc_score = auc(y_pred, y_true) 
        prf_score = prf(y_pred, y_true)
        conf_matrix = confusion(y_pred, y_true)

        accs.append(bal_acc)
        aucs.append(auc_score)
        prfs.append(prf_score)
        confus.append(conf_matrix)
        
        print(f'Fold {fold+1} - Balanced Accuracy: {bal_acc:.4f}, AUC: {auc_score:.4f}')

    # Fold 평균 출력
    print(f'Average Balanced Accuracy: {np.mean(accs):.4f}')
    print(f'Average AUC: {np.mean(aucs):.4f}')
    print(f'Average PRF: {np.mean(prfs, axis=0)}')
    print(f'Average Confusion Matrix: {np.mean(confus, axis=0)}')
    print(train_time)

    print("\r\n========================== Finish ==========================") 
    n_samples = raw_features.shape[0]
    acc_nfold = np.mean(accs)
    acc_nfold_std = np.std(accs)
    auc_nfold = np.mean(aucs)
    auc_nfold_std = np.std(aucs)
    
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

    if not os.path.exists('./results+DNN+batch.csv'):
        df.to_csv('./results+DNN+batch.csv', index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv('./results+DNN+batch.csv', index=False, mode='a', encoding='utf-8-sig', header=False)



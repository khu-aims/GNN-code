import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from opt import * 
from utils.metrics import balanced_accuracy, prf_ml, auc_ml, confusion
from dataloader import dataloader 
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time

now = datetime.now()
dt_string = now.strftime("%m_%d_%H_%M")
train_time = []
if __name__ == '__main__':
    opt = OptInit().initialize()

    print('  Loading dataset ...')
    dl = dataloader() 
    raw_features, y, nonimg, site = dl.load_data(opt)

    parameters = {'C':[0.1, 1,10],'gamma':[0.1, 1, 10]}

    n_folds = opt.n_folds
    cv_splits = dl.data_split(n_folds)
    nested_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    accs = np.zeros(n_folds, dtype=np.float32) 
    aucs = np.zeros(n_folds, dtype=np.float32)
    prfs = np.zeros([n_folds,3], dtype=np.float32)
    confus = np.zeros([n_folds,2], dtype=np.float32)

    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold)) 

        train_ind = cv_splits[fold][0] 
        test_ind = cv_splits[fold][1] 

        scaler = StandardScaler()
        x_train = scaler.fit_transform(raw_features[train_ind])
        x_test = scaler.transform(raw_features[test_ind])

        print(x_train.shape)
        
        model = SVC(kernel='linear',random_state=123)
        grid = GridSearchCV(model, parameters, cv=nested_skf, scoring='balanced_accuracy')
        grid.fit(x_train, y[train_ind])
        print('    The best parameter C: %.2e with BAC of %f' % (grid.best_params_['C'], grid.best_score_))
        start_time = time.time()
        clf = SVC(kernel='linear', C=grid.best_params_['C'], gamma=grid.best_params_['gamma'], 
                  probability=True, random_state=123)
        clf.fit(x_train, y[train_ind])
        end_time = time.time()
        train_time.append(end_time - start_time) 
        
        logits_test = clf.predict(x_test)
        logits_test_proba = clf.predict_proba(x_test)[:,1]

        tn, fp, fn, tp = confusion_matrix(y[test_ind], logits_test).ravel()
        fold_sen = tp / (tp + fn)
        fold_spe = tn / (tn + fp)
        accs[fold] = (fold_sen + fold_spe) / 2
        aucs[fold] = auc_ml(logits_test_proba,y[test_ind]) 
        prfs[fold]  = prf_ml(logits_test,y[test_ind])  
        confus[fold]  = [fold_sen, fold_spe]

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

    arg_dict = vars(opt)
    arg_dict["ML"] = "SVM"
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
    arg_dict["train_time_mean"] = sum(train_time)/5
    df = pd.DataFrame(arg_dict)
    
    if not os.path.exists('results+ML.csv'):
        df.to_csv('results+ML.csv', index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv('results+ML.csv', index=False, mode='a', encoding='utf-8-sig', header=False)


import numpy as np 
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, balanced_accuracy_score
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

def balanced_accuracy(preds, labels):
    """Accuracy with masking. Acc of the masked samples"""
    bac = balanced_accuracy_score(labels, np.argmax(preds, 1)).astype(np.float32)
    return bac

def auc(preds, labels, is_logit=True):
    ''' input: logits, labels  ''' 
    if is_logit:
        pos_probs = softmax(preds, axis=1)[:, 1]
    else:
        pos_probs = preds[:,1]
    try:
        auc_out = roc_auc_score(labels, pos_probs)
    except:
        auc_out = 0
    return auc_out

def prf(preds, labels):
    ''' input: logits, labels  ''' 
    pred_lab= np.argmax(preds, 1)
    p,r,f,s  = precision_recall_fscore_support(labels, pred_lab, average='binary')
    return [p,r,f]

def confusion(preds, labels):
    pred_lab= np.argmax(preds, 1)
    
    tn, fp, fn, tp = confusion_matrix(labels, pred_lab).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)

    return [sen, spe]

def prf_ml(preds, labels, is_logit=True):
    ''' input: logits, labels  ''' 
    p,r,f,s  = precision_recall_fscore_support(labels, preds, average='binary')
    return [p,r,f]


def auc_ml(preds, labels, is_logit=True):
    try:
        auc_out = roc_auc_score(labels, preds)
    except:
        auc_out = 0
    return auc_out


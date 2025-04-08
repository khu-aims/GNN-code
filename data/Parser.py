import os
import csv
import numpy as np
import scipy.io as sio

from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE,SelectKBest, f_classif
from scipy.spatial import distance
import bct
from sklearn.decomposition import PCA
from scipy.io import loadmat
from nctpy.metrics import ave_control
from nctpy.utils import (
    matrix_normalization,
    convert_states_str2int,
    normalize_state,
    normalize_weights,
    get_null_p,
    get_fdr_p,
)

# Input data variables
root_folder = './data/SZ_MSN'
data_folder = root_folder
phenotype = os.path.join(root_folder, 'phenotype.csv')

# Get the list of subject IDs
def get_ids(num_subjects=None):
    subject_IDs = np.genfromtxt(os.path.join(data_folder, 'subject_IDs.txt'), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs

# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]
    return scores_dict

# Dimensionality reduction step for the feature vector using a ridge classifier
def feature_selection(features, labels, train_ind, fnum):

    estimator = RidgeClassifier()
    selector = RFE(estimator, n_features_to_select=fnum, step=100, verbose=0)

    featureX = features[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(features)

    return x_data

def pca_decomposition(features, labels, train_ind, fnum):

    pca = PCA(n_components=fnum)
    featureX = features[train_ind, :]
    featureY = labels[train_ind]

    selector = pca.fit(featureX)
    x_data = selector.transform(features)

    return x_data


def select_KBest(features, labels, train_ind, fnum):
    selector = SelectKBest(f_classif, k=fnum)

    featureX = features[train_ind, :]
    featureY = labels[train_ind]
    selector = selector.fit(featureX, featureY.ravel())
    x_data = selector.transform(features)

    return x_data


def get_networks(subject_list, variable='connectivity', node = "Wei-CC", msn_threshold = 0.3):
    all_networks = []
    
    for subject in subject_list:
        fl = os.path.join(data_folder, "corr_" + subject + ".mat")
        matrix = sio.loadmat(fl)[variable]
        np.fill_diagonal(matrix, 0)

        if node == "Wei-Betweenness" and msn_threshold == 1.0: # Betweenness takes more time than other centralities
            bet = sio.loadmat(fl)['Betweenness'][0]
            #bet = sio.loadmat(fl)['10'][0]
            all_networks.append(bet)
            
        elif node == "Wei-Betweenness" and msn_threshold < 1.0: # Betweenness takes more time than other centralities
            #bet = sio.loadmat(fl)[f'Betweenness_{str(int(msn_threshold*10))}'][0]
            bet = sio.loadmat(fl)[f'{str(int(msn_threshold*10))}'][0]
            all_networks.append(bet)
        
        elif node== "Wei-CC":
            thres_df = bct.threshold_proportional(matrix, msn_threshold, copy=True)
            cc = bct.clustering_coef_wu(thres_df)
            all_networks.append(cc)
        elif node== "Wei-Eigen":
            thres_df = bct.threshold_proportional(matrix, msn_threshold, copy=True)
            eigen = bct.eigenvector_centrality_und(thres_df)
            all_networks.append(eigen)
        elif node == "Wei-Strength":
            thres_df = bct.threshold_proportional(matrix, msn_threshold, copy=True)
            strs = bct.strengths_und(thres_df)
            all_networks.append(strs)   
        elif node == "Wei-Absstrength":
            thres_df = bct.threshold_proportional(abs(matrix), msn_threshold, copy=True)
            strs = bct.strengths_und(thres_df)
            all_networks.append(strs)    

        elif node== "Wei-Concat":
            thres_df = bct.threshold_proportional(matrix, msn_threshold, copy=True)
            strs = bct.strengths_und(thres_df)

            if msn_threshold == 1.0:
                bet = sio.loadmat(fl)['Betweenness'][0]
                #bet = sio.loadmat(fl)['10'][0]

            elif msn_threshold < 1.0:
                bet = sio.loadmat(fl)[f'Betweenness_{str(msn_threshold)}'][0]
            
            eigen = bct.eigenvector_centrality_und(thres_df)
            cc = bct.clustering_coef_wu(thres_df)
            merged_values = np.concatenate((strs, bet, eigen, cc))
            all_networks.append(merged_values)
        else: # When using KBest or RFE
            all_networks.append(matrix)
    
    if node in ["Wei-Betweenness", "Wei-Strength", "Wei-CC", "Wei-Absstrength", "Wei-Eigen", "Wei-Concat"] :
        all_networks = np.array(all_networks)
        matrix = np.vstack(all_networks)

    else: # When using KBest or RFE
        idx = np.triu_indices_from(all_networks[0], 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_networks = [np.arctanh(mat) for mat in all_networks]
        vec_networks = [mat[idx] for mat in norm_networks]
        matrix = np.vstack(vec_networks)

    return matrix

def create_affinity_graph_from_scores(scores, pd_dict):
    num_nodes = len(pd_dict[scores[0]]) 
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = pd_dict[l]
        std = np.std(label_dict)

        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < std:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if np.array_equal(label_dict[k], label_dict[j]):
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph

def get_static_affinity_adj(features, pd_dict, static):
    pd_affinity = create_affinity_graph_from_scores(static, pd_dict)
    distv = distance.pdist(features, metric='correlation') 
    dist = distance.squareform(distv)  
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
    adj = pd_affinity * feature_sim  

    return adj
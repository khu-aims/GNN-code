import data.Parser as Reader
import numpy as np
from utils.gcn_utils import preprocess_features
from sklearn.model_selection import StratifiedKFold
import category_encoders as ce
import pandas as pd
from sklearn.model_selection import GroupKFold

class dataloader():
    def __init__(self): 
        self.pd_dict = {}
        self.num_classes = 2 

    def load_data(self, opt):
        subject_IDs = Reader.get_ids()
        labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
        num_nodes = len(subject_IDs)

        sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
        unique = np.unique(list(sites.values())).tolist()
        ages = Reader.get_subject_score(subject_IDs, score='AGE_AT_SCAN')
        genders = Reader.get_subject_score(subject_IDs, score='SEX') 

        y_onehot = np.zeros([num_nodes, self.num_classes])
        y = np.zeros([num_nodes])
        site = np.zeros([num_nodes], dtype=np.int32)
        age = np.zeros([num_nodes], dtype=np.float32)
        gender = np.zeros([num_nodes], dtype=np.int32)

        for i in range(num_nodes):
            y_onehot[i, int(labels[subject_IDs[i]])-1] = 1
            y[i] = int(labels[subject_IDs[i]])
            site[i] = unique.index(sites[subject_IDs[i]])
            age[i] = float(ages[subject_IDs[i]])
            gender[i] = genders[subject_IDs[i]] 
        
        self.y = y - 1  #If y(class) starts from 1, it must be set.
        self.raw_features = Reader.get_networks(subject_IDs, "connectivity", opt.node, opt.msn_threshold)
        
        df = pd.DataFrame({"site": site, "gender" : gender})

        if opt.use_encoding == "onehot":
            encoder = ce.OneHotEncoder(cols=["site", "gender"])
            encoded_df = encoder.fit_transform(df)
            site_df = encoded_df.filter(regex='^site')
            sex_df = encoded_df.filter(regex='^gender')
            age_df = pd.DataFrame(age, columns=['AGE_AT_SCAN'])
            # Concatenate along columns
            phonetic_data = pd.concat([encoded_df, age_df], axis=1).to_numpy()

            self.pd_dict['SITE_ID'] = np.copy(site_df.to_numpy())
            self.pd_dict['SEX'] = np.copy(sex_df.to_numpy()) 
            self.pd_dict['AGE_AT_SCAN'] = np.copy(age)

        else:
            phonetic_data = np.zeros([num_nodes, 3], dtype=np.float32)
            phonetic_data[:,0] = site
            phonetic_data[:,1] = gender - 1
            phonetic_data[:,2] = age 

            self.pd_dict['SITE_ID'] = np.copy(site)
            self.pd_dict['SEX'] = np.copy(gender - 1)
            self.pd_dict['AGE_AT_SCAN'] = np.copy(age)

        self.site = site

        return self.raw_features, self.y, phonetic_data, self.site


    def create_stratification_key(self):
        return [str(site_val) + '_' +  str(y_val) for y_val, site_val in zip(self.y, self.site)]

    def data_split(self, n_folds):
        # split data by k-fold CV
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=123)
        cv_splits = list(skf.split(self.raw_features, self.y))
        return cv_splits 

    def data_split_loso(self):
        # split data by leave-one-site-out CV
        groups = self.site  # SITE_ID acts as the group identifier
        gkf = GroupKFold(n_splits=len(np.unique(groups)))
        cv_splits = list(gkf.split(self.raw_features, self.y, groups=groups))
        return cv_splits

    def get_node_features(self, train_ind, node, node_ftr_dim):
        if node == 'kBest':
            node_ftr = Reader.select_KBest(self.raw_features, self.y, train_ind, node_ftr_dim)
        elif node == 'RFE':
            node_ftr = Reader.feature_selection(self.raw_features, self.y, train_ind,node_ftr_dim) 
        elif node == 'PCA':
            node_ftr = Reader.pca_decomposition(self.raw_features, self.y, train_ind,node_ftr_dim) 
        elif node in ["Wei-Betweenness", "Wei-Strength", "Wei-CC", "Wei-Absstrength", "Wei-Eigen", "Wei-Concat"] :
            node_ftr = self.raw_features
            
        self.node_ftr = preprocess_features(node_ftr) 
        return self.node_ftr

    def get_ENCODER_inputs(self, node_ftr, static, quantile):
        # construct edge network inputs 
        n = self.node_ftr.shape[0] 
        num_edge = n*(1+n)//2 - n  
        pd_ftr_dim = node_ftr.shape[1]
        edge_index = np.zeros([2, num_edge], dtype=np.int64) 
        edgenet_input = np.zeros([num_edge, 2*pd_ftr_dim], dtype=np.float32)  
        aff_score = np.zeros(num_edge, dtype=np.float32)   
        # static affinity score used to pre-prune edges 
        aff_adj = Reader.get_static_affinity_adj(self.node_ftr, self.pd_dict, static)  
        flatten_ind = 0 
        for i in range(n):
            for j in range(i+1, n):
                edge_index[:,flatten_ind] = [i,j]
                edgenet_input[flatten_ind]  = np.concatenate((node_ftr[i], node_ftr[j]))
                aff_score[flatten_ind] = aff_adj[i][j]  
                flatten_ind +=1

        thres =  np.quantile(aff_score, quantile)
        assert flatten_ind == num_edge, "Error in computing edge input"
        keep_ind = np.where(aff_score > thres)[0]  
        edge_index = edge_index[:, keep_ind]
        edgenet_input = edgenet_input[keep_ind]

        return edge_index, edgenet_input, thres

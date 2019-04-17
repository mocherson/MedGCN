
import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
from sklearn.model_selection import ShuffleSplit
from utility.preprocessing import *


path = '~/MedGCN/data/'
enc_pat = pd.read_csv(path+'inter_enc_pat.csv',index_col=0)
enc_lab = pd.read_csv(path+'inter_enc_lab.csv',index_col=0)
enc_med = pd.read_csv(path+'inter_enc_med.csv',index_col=0)

enc_lab = maxminnorm(enc_lab)

assert (enc_pat.index==enc_lab.index).all() and (enc_pat.index==enc_med.index).all(), 'make sure the enccounter are in the same order.'


n_enc, n_pat = enc_pat.shape
_, n_lab = enc_lab.shape
_, n_med = enc_med.shape

enc_pat_adj = torch.FloatTensor(enc_pat.values).to_sparse().cuda()
enc_lab_adj = sparse_to_tensor(enc_lab.to_sparse().to_coo()).cuda()
enc_med_adj = torch.FloatTensor(enc_med.astype(bool).astype(np.uint8).values).to_sparse().cuda()

enc_feat = sparse_to_tensor(sp.identity(n_enc)).cuda()
pat_feat = sparse_to_tensor(sp.identity(n_pat)).cuda()
lab_feat = sparse_to_tensor(sp.identity(n_lab)).cuda()
med_feat = sparse_to_tensor(sp.identity(n_med)).cuda()

adj_mats={(0,1): [enc_pat_adj],
          (0,2): [enc_lab_adj],
          (0,3): [enc_med_adj],
         }
fea_mats={0: enc_feat, 1: pat_feat, 2: lab_feat, 3: med_feat}
fea_num = {0: n_enc, 1: n_pat, 2: n_lab, 3: n_med}

adj_losstype = { (0, 1): [('BCE',0)],   (0, 2): [('MSE',1)],   (0, 3): [('BCE',1)] }


# medicine recommendation
ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
ss1 = ShuffleSplit(n_splits=1, test_size=0.1, random_state=12345)
med_train_val_idx, med_test_idx =next(ss.split(enc_med))
train_idx, val_idx =next(ss1.split(med_train_val_idx))
med_train_idx, med_val_idx = med_train_val_idx[train_idx], med_train_val_idx[val_idx]

train_mask_enc_med=torch.zeros(enc_med_adj.shape)
train_mask_enc_med[med_train_idx]=1
val_mask_enc_med=torch.zeros(enc_med_adj.shape)
val_mask_enc_med[med_val_idx]=1
test_mask_enc_med=torch.zeros(enc_med_adj.shape)
test_mask_enc_med[med_test_idx]=1

# train_adj_masks = {   (0, 3): [train_mask_enc_med.cuda()] }
# val_adj_masks = {    (0, 3): [val_mask_enc_med.cuda()] }
# test_adj_masks = {    (0, 3): [test_mask_enc_med.cuda()] }
# edge_decoder = {    (0, 3): ('linear',1) }
tasks=((0,3), (0,2))


# # lab estimation,  matrix completion
ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
ss1 = ShuffleSplit(n_splits=1, test_size=0.1, random_state=12345)
lab_train_val_idx, lab_test_idx =next(ss.split(enc_lab_adj._values()))
train_idx, val_idx =next(ss1.split(lab_train_val_idx))
lab_train_idx, lab_val_idx = lab_train_val_idx[train_idx], lab_train_val_idx[val_idx]

indices = enc_lab_adj._indices()
mask_enc_lab = torch.sparse.FloatTensor(indices, torch.ones(indices.shape[1]).cuda(), enc_lab_adj.shape)
train_mask_enc_lab = torch.sparse.FloatTensor(indices[:,lab_train_idx], torch.ones(len(lab_train_idx)).cuda(), enc_lab_adj.shape)
val_mask_enc_lab = torch.sparse.FloatTensor(indices[:,lab_val_idx], torch.ones(len(lab_val_idx)).cuda(), enc_lab_adj.shape)
test_mask_enc_lab = torch.sparse.FloatTensor(indices[:,lab_test_idx], torch.ones(len(lab_test_idx)).cuda(), enc_lab_adj.shape)

# train_adj_masks = {   (0, 2): [train_mask_enc_lab.cuda()] }
# val_adj_masks = {    (0, 2): [val_mask_enc_lab.cuda()] }
# test_adj_masks = {    (0, 2): [test_mask_enc_lab.cuda()] }
# edge_decoder = {    (0, 2): ('distmult',1) }


# # both
train_adj_masks = {   (0, 2): [train_mask_enc_lab.cuda()], (0, 3): [train_mask_enc_med.cuda()] }
val_adj_masks = {    (0, 2): [val_mask_enc_lab.cuda()], (0, 3): [val_mask_enc_med.cuda()]  }
test_adj_masks = {    (0, 2): [test_mask_enc_lab.cuda()] , (0, 3): [test_mask_enc_med.cuda()]}
# edge_decoder = {    (0, 2): ('distmult',1),   (0, 3): ('distmult',1) }




# train_adj_masks = { (0, 2): [mask_enc_lab.cuda()], (0, 3): [train_mask_enc_med.cuda()] }
# edge_decoder = { (0, 1): ('distmult',1),    (0, 2): ('distmult',1),   (0, 3): ('distmult',1) }


## inductive learning
# train_adj_mats={(0,1): [to_dense(enc_pat_adj)[med_train_val_idx]],
#           (0,2): [to_dense(enc_lab_adj)[med_train_val_idx]],
#           (0,3): [to_dense(enc_med_adj)[med_train_val_idx]],
#          }

# masks = train_adj_masks

# train_adj_masks = {   (0, 2): [to_dense(train_mask_enc_lab)[med_train_val_idx].cuda()], (0, 3): [train_mask_enc_med[med_train_val_idx].cuda()] }

# train_fea_mats={0: to_dense(enc_feat)[med_train_val_idx], 1: pat_feat, 2: lab_feat, 3: med_feat}



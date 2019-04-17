import torch

from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import scipy.sparse as sp
from utility.preprocessing import sparse_to_tensor, to_dense
from utility.rank_metrics import *
from sklearn.metrics import label_ranking_average_precision_score, mean_squared_error

from model import MedGAE, MultiMatLoss 

from preparedata import * 

epochs=200

# n_enc = 500   # node type 0
# n_lab = 100   # node type 1
# n_med = 50    # node type 2

# enc_enc_adj = sparse_to_tensor(sp.csr_matrix((10 * np.random.randn(n_enc, n_enc) > 15).astype(float))).cuda()
# enc_lab_adj = torch.Tensor(np.random.rand(n_enc, n_lab)).cuda()
# enc_med_adj = sparse_to_tensor(sp.csr_matrix((10 * np.random.randn(n_enc, n_med) > 20).astype(float))).cuda()

# enc_feat = sparse_to_tensor(sp.identity(n_enc)).cuda()
# lab_feat = sparse_to_tensor(sp.identity(n_lab)).cuda()
# med_feat = sparse_to_tensor(sp.identity(n_med)).cuda()

# adj_mats={(0,0): [enc_enc_adj + enc_enc_adj.t()],
#           (0,1): [enc_lab_adj],
#           (0,2): [enc_med_adj],
#          }
# fea_mats={0: enc_feat, 1: lab_feat, 2: med_feat}
# fea_num = {0: n_enc, 1: n_lab, 2: n_med}

# edge_decoder = {
#     (0, 0): ('distmult',1),
#     (0, 1): ('distmult',1),
#     (0, 2): ('distmult',1)
# }
# adj_masks = {
#     (0, 0): [torch.randint(0,2, enc_enc_adj.shape).cuda()],
#     (0, 1): [torch.randint(0,2, enc_lab_adj.shape).cuda()],
#     (0, 2): [torch.randint(0,2, enc_med_adj.shape).cuda()]
# }

# test_adj_masks = {
#     (0, 0): [torch.randint(0,2, enc_enc_adj.shape).cuda()],
#     (0, 1): [torch.randint(0,2, enc_lab_adj.shape).cuda()],
#     (0, 2): [torch.randint(0,2, enc_med_adj.shape).cuda()]
# }


medgcn =  MedGAE(fea_num,( 200,  ), edge_decoder, dropout=0).cuda()
optimizer = optim.Adam(medgcn.parameters(), lr=1e-3)
lossfun = MultiMatLoss().cuda()
# scheduler = ReduceLROnPlateau(optimizer, 'min')

# training-validation-test
tr_loss, v_loss = [],[]
v_mapk, te_mapk = [],[]
v_lrap, te_lrap = [],[]
v_mse, te_mse = [],[]
for epoch in range(epochs):
    #training
    medgcn.train()
    optimizer.zero_grad()
    adj_recon, z = medgcn(fea_mats, adj_mats, train_adj_masks )
    train_loss=lossfun(adj_recon, adj_mats, train_adj_masks, adj_losstype)
    train_loss.backward()
    optimizer.step()
    
    # validation-test
    medgcn.eval()
    with torch.no_grad():
        adj_recon, z = medgcn(fea_mats, adj_mats, train_adj_masks )
        val_loss = lossfun(adj_recon, adj_mats, val_adj_masks, adj_losstype)
        test_loss = lossfun(adj_recon, adj_mats, test_adj_masks, adj_losstype)
        # scheduler.step(val_loss)
        if (0,3) in adj_recon:
            actual = to_dense(adj_mats[(0,3)][0]).cpu().numpy()
            predicted = adj_recon[(0,3)][0].cpu().numpy()
            val_mapk = mapk(actual[med_val_idx], predicted[med_val_idx], k=2)
            val_lrap = label_ranking_average_precision_score(actual[med_val_idx], predicted[med_val_idx])
            test_mapk = mapk(actual[med_test_idx], predicted[med_test_idx], k=2)
            test_lrap = label_ranking_average_precision_score(actual[med_test_idx], predicted[med_test_idx])
            train_mapk = mapk(actual[med_train_idx], predicted[med_train_idx], k=2)
            train_lrap = label_ranking_average_precision_score(actual[med_train_idx], predicted[med_train_idx])
        if (0,2) in adj_recon:
            pred = nn.Sigmoid()(adj_recon[(0,2)][0]) if adj_losstype[(0,2)] =='BCE' else adj_recon[(0,2)][0]
            idx = adj_mats[(0,2)][0]._indices()
            predicted = pred[idx[0],idx[1]].cpu().numpy()
            actual = adj_mats[(0,2)][0]._values().cpu().numpy()
            val_mse = mean_squared_error(predicted[lab_val_idx], actual[lab_val_idx])
            test_mse = mean_squared_error(predicted[lab_test_idx], actual[lab_test_idx])
                
    print('====> Epoch: {} train loss: {:.4f}, validation loss: {:.4f}, test loss: {:.4f}'  \
          .format(epoch, train_loss.item(), val_loss.item(), test_loss.item()))
    tr_loss.append(train_loss.item())
    v_loss.append(val_loss.item())
    if (0,3) in adj_recon:   
        print('medicine recommendation: validation map@2: {:.4f}, test map@2: {:.4f},  validation LRAP: {:.4f}, test LRAP: {:.4f} \n  \
        train map@2: {:.4f},  train LRAP: {:.4f}'  \
          .format( val_mapk, test_mapk, val_lrap, test_lrap, train_mapk, train_lrap) )
        v_mapk.append(val_mapk)
        te_mapk.append(test_mapk)
        v_lrap.append(val_lrap)
        te_lrap.append(test_lrap)
    if (0,2) in adj_recon:
        print('lab estimation: validation mse: {:.4f}, test mse: {:.4f}'.format( val_mse, test_mse) )
        v_mse.append(val_mse)
        te_mse.append(test_mse)
    
res=pd.DataFrame({   'tr_loss': tr_loss, 'v_loss': v_loss, 
                  'v_mapk': v_mapk,'te_mapk': te_mapk, 'v_lrap': v_lrap, 'te_lrap': te_lrap,
                  # 'v_mse': v_mse, 'te_mse': te_mse
                 })
# res.to_csv('res_lab_BCE.csv')



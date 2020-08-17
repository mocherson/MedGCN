import argparse

import torch

from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import scipy.sparse as sp
from utility.preprocessing import sparse_to_tensor, to_dense
from utility.rank_metrics import *
from utility.iofile import *
from sklearn.metrics import label_ranking_average_precision_score, mean_squared_error

from model import MedGCN, MultiMatLoss 

from preparedata import * 

parser = argparse.ArgumentParser(description='Training and evaluating the MedGCN')
parser.add_argument('--lamb', type=float, default=1, help='regularization parameter for lab error')
parser.add_argument('--layers', type=int, default=1, help='GCN layers')
args = parser.parse_args()

torch.manual_seed(123)

epochs=300
# n_iter_no_change=50

adj_losstype = { (0, 1): [('BCE',0)],   (0, 2): [('MSE',args.lamb)],   (0, 3): [('BCE',1)] }

medgcn =  MedGCN(fea_num,( 300,  )*args.layers, tasks, dropout=0.0).cuda()
optimizer = optim.Adam(medgcn.parameters(), lr=1e-3)
lossfun = MultiMatLoss(pos_weight=None).cuda()
# scheduler = ReduceLROnPlateau(optimizer, 'min')

# training-validation-test
tr_loss, v_loss = [],[]
v_mapk, te_mapk = [],[]
v_lrap, te_lrap = [],[]
v_mse, te_mse = [],[]
best_val_loss = np.inf
best_val_lrap, best_val_mse = 0, np.inf
best_epoch = 0
for epoch in range(epochs):
    #training
    print("training............")
    medgcn.train()
    optimizer.zero_grad()
    adj_recon, z = medgcn(fea_mats, adj_mats, train_adj_masks )
    train_loss=lossfun(adj_recon, adj_mats, train_adj_masks, adj_losstype)
    train_loss.backward()
    optimizer.step()
    
    # validation-test
    print("validation............")
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
            pred = adj_recon[(0,2)][0]
            idx = adj_mats[(0,2)][0]._indices()
            predicted = pred[idx[0],idx[1]].cpu().numpy()
            actual = adj_mats[(0,2)][0]._values().cpu().numpy()
            # val_mse = mean_squared_error(predicted, actual)
            # test_mse = mean_squared_error(predicted, actual)
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
        
#     if val_loss<=best_val_loss:
#         best_val_loss=val_loss
#         best_epoch=epoch
        
#     if test_mse<=best_val_mse:
#         best_val_mse=test_mse
#         save_obj(adj_recon[(0,2)][0].cpu().numpy(), 'est_lab2.pkl')
#         best_epoch=epoch
        
#     if test_lrap>=best_val_lrap:
#         best_val_lrap=test_lrap
#         save_obj(adj_recon[(0,3)][0].cpu().numpy(), 'rec_med2.pkl')
#         best_epoch=epoch

#     if epoch-best_epoch>n_iter_no_change:
#         break
    
res=pd.DataFrame({   'tr_loss': tr_loss, 'v_loss': v_loss, 
                  'v_mapk': v_mapk,'te_mapk': te_mapk, 'v_lrap': v_lrap, 'te_lrap': te_lrap,
                  'v_mse': v_mse, 'te_mse': te_mse
                 })
# res.to_csv('res_Med_lab%s.csv'%(args.lamb))

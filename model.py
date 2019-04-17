import scipy.sparse as sp
import copy
import torch

from torch import nn
import torch.nn.functional as F

from layers import *

from utility.preprocessing import str2value, issymmetric, to_dense, to_sparse




class HGCN(nn.Module):
    def __init__(self, in_dim, out_dim, selfweight=1):
        super(HGCN, self).__init__()
        self.selfweight = selfweight
        if isinstance(in_dim, dict):
            self.gcn = nn.ModuleDict({str(i): GraphConvolution(in_dim[i],out_dim) for i in in_dim})
            self.ishg = True
        elif isinstance(in_dim, int):
            self.gcn = GraphConvolution(in_dim,out_dim)
            self.ishg = False
        else:
            raise ValueError('in_dim must be integer or dict of integer')
    
    def forward(self, fea_mats, adj_mats ):
        gcn = self.gcn if self.ishg else {str(i):self.gcn for i in fea_mats} 
        out_fea = {i: self.selfweight*gcn[str(i)](fea_mats[i]) for i in fea_mats}        
        for (i, j), adjs in adj_mats.items():
            for adj in adjs:
                out_fea[i] = out_fea[i] + gcn[str(j)](fea_mats[j], adj)
                if i!=j or not issymmetric(adj) :
                    out_fea[j] = out_fea[j] + gcn[str(i)](fea_mats[i], adj.t())
            
        return out_fea
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + self.gcn.__repr__()+')'

class MedGCN(nn.Module):
    def __init__(self, in_dim,  out_dims, tasks=((0,3),), dropout=0.5):
        super(MedGCN, self).__init__()
        self.encoder = nn.ModuleList( [HGCN(in_dim, out_dims[0])] )
        self.Dropout = DictDropout(p=dropout)
        self.tasks = tasks
        self.predicter = nn.ModuleDict( {str(i): nn.Linear(out_dims[-1], in_dim[i[1]]) for i in tasks })
        for i in range(len(out_dims)):
            if i+1<len(out_dims):
                self.encoder.append( HGCN(out_dims[i], out_dims[i+1]) )   
    
    def encode(self, fea_mats, adj_mats ):
        for i, m in enumerate(self.encoder):
            fea_mats = m(fea_mats, adj_mats)
            if i+1<len(self.encoder):
                fea_mats = DictReLU()(fea_mats)
                fea_mats = self.Dropout(fea_mats)

        return fea_mats
    
    def predict(self, z):
        z = DictReLU()(z)
        z = self.Dropout(z)
        adj_recon={}
        for t in self.tasks:
            adj_recon[t] = [self.predicter[str(t)](z[t[0]])]
            
        return adj_recon
    
    def forward(self, fea_mats, adj_mats, adj_masks):
        adj_mats = copy.deepcopy(adj_mats)
        for key in adj_masks.keys() & adj_mats.keys():
            adj_mats[key] = [to_sparse(adj_masks[key][i]).float()*to_sparse(adj_mats[key][i]) for i in range(len(adj_masks[key]))]
            # if key in self.tasks:
            #     adj_mats.pop(key)
            
        z = self.encode(fea_mats, adj_mats)
        adj_recon = self.predict(z)
        return adj_recon, z    

class MedGAE(nn.Module):
    def __init__(self, in_dim,  out_dims, edge_decoder, dropout=0.5):
        super(MedGAE, self).__init__()
        self.encoder = nn.ModuleList( [HGCN(in_dim, out_dims[0])] )
        for i in range(len(out_dims)):
            if i+1<len(out_dims):
                self.encoder.append( HGCN(out_dims[i], out_dims[i+1]) )

        self.decoder = nn.ModuleDict({})
        for nodes, (dec, num) in edge_decoder.items():
            if dec == 'innerproduct':
                decoder = InnerProductDecoder(out_dims[-1], num)
            elif dec == 'distmult':
                decoder = DistMultDecoder(out_dims[-1], num)
            elif dec == 'bilinear':
                decoder = BilinearDecoder(out_dims[-1], num, issymmetric=False)
            elif dec == 'dedicom':
                decoder = DEDICOMDecoder(out_dims[-1], num, issymmetric=False)
            elif dec == 'symbilinear':
                decoder = BilinearDecoder(out_dims[-1], num, issymmetric=True)
            elif dec == 'symdedicom':
                decoder = DEDICOMDecoder(out_dims[-1], num, issymmetric=True)
            elif dec == 'symMLP':
                decoder = MLPDecoder(out_dims[-1], num, issymmetric=True)
            elif dec == 'MLP':
                decoder = MLPDecoder(out_dims[-1], num, issymmetric=False)
            elif dec == 'symlinear':
                decoder = LinearDecoder(out_dims[-1], num, issymmetric=True)
            elif dec == 'linear':
                decoder = LinearDecoder(out_dims[-1], num, issymmetric=False)    
            else:
                raise ValueError('Unknown decoder type')
            self.decoder[str(nodes)] = decoder
            self.dropout = dropout
    
    def encode(self, fea_mats, adj_mats ):
        for i, m in enumerate(self.encoder):
            fea_mats = m(fea_mats, adj_mats)
            if i+1<len(self.encoder):
                fea_mats = DictReLU()(fea_mats)
                fea_mats = DictDropout(p=self.dropout)(fea_mats)

        return fea_mats
    
    def decode(self, z):
        adj_recon={}
        for nodes, decoder in self.decoder.items():
            nodes = str2value(nodes)            
            adj_recon[nodes] = [decoder(z[nodes[0]], z[nodes[1]], i) for i in range(decoder.num_types)]
            
        return adj_recon
    
    def forward(self, fea_mats, adj_mats, adj_masks):
        adj_mats = copy.deepcopy(adj_mats)
        for key, masks in  adj_masks.items():
            adj_mats[key] = [to_sparse(masks[i]).float()*to_sparse(adj_mats[key][i]) for i in range(len(masks))]
            
        z = self.encode(fea_mats, adj_mats)
        adj_recon = self.decode(z)
        return adj_recon, z
    
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MatWithNALoss(nn.Module):
    def __init__(self, reduction='mean', pos_weight=None):
        super(MatWithNALoss, self).__init__()
        self.reduction  = reduction
        self.pos_weight = pos_weight
        
    def forward(self, input, target, mask, losstype= None ):
        # losstype in {'BCE', 'MSE'}
        input = input.view(-1)
        target = target.view(-1)
        if not isinstance(mask, (int,float)):
            mask = to_dense(mask.byte()).view(-1)
            target = target[mask]
            input = input[mask]
        pos_weight= self.pos_weight if self.pos_weight else (target==0).sum()/ (target!=0).sum()
        if not losstype: 
            losstype = 'BCE' if (target>=0).all() and (target<=1).all() else 'MSE' 
        
        if losstype=='BCE':
            loss = F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction, pos_weight=pos_weight)
        elif losstype=='MSE':
            loss = F.mse_loss(input, target, reduction=self.reduction)
        elif losstype=='L1':
            loss = F.l1_loss(input, target, reduction=self.reduction)
        else:
            raise ValueError('Undefined loss type.')
        # print("using loss type "+losstype+':'+str(loss.item()))
        return loss
    
class MultiMatLoss(nn.Module):
    def __init__(self, reduction='mean', pos_weight=None):
        super(MultiMatLoss, self).__init__()
        self.losscls = MatWithNALoss(pos_weight = pos_weight)
        self.reduction  = reduction
        
        
    def forward(self, adj_recon, adj_mats, adj_masks, adj_losstype=None ):
        loss=0
        for key, adj in adj_recon.items():
            # print('computing loss for type:'+str(key))
            for i in range(len(adj)):
                input = adj_recon[key][i]
                target = to_dense(adj_mats[key][i])                
                mask = adj_masks[key][i].byte() if key in adj_masks else 1
                losstype = adj_losstype[key][i] if isinstance(adj_losstype, dict) else adj_losstype
                loss += losstype[1]*self.losscls(input, target, mask, losstype[0])
        return loss
 
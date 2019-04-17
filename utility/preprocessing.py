import numpy as np
import scipy.sparse as sp
import torch


def get_sparse_mat(a2b, a2idx, b2idx):
    n = len(a2idx)
    m = len(b2idx)
    assoc = np.zeros((n, m))
    for a, b_assoc in a2b.iteritems():
        if a not in a2idx:
            continue
        for b in b_assoc:
            if b not in b2idx:
                continue
            assoc[a2idx[a], b2idx[b]] = 1.
    assoc = sp.coo_matrix(assoc)
    return assoc


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def sparse_to_tensor(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = torch.LongTensor(np.vstack((sparse_mx.row, sparse_mx.col)))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(coords, values, shape)

def to_sparse(x):
    return x if x.is_sparse else x.to_sparse()

def to_dense(x):
    return x if not x.is_sparse else x.to_dense()

def str2value(code):
    exec('global i; i = %s' % code)
    global i
    return i

def issymmetric(mat):
    if torch.is_tensor(mat):
        mat = mat.to_dense().cpu().numpy() if mat.is_sparse else mat.cpu().numpy()
    if mat.shape!=mat.T.shape:
        return False
    if sp.issparse(mat):
        return not (mat!=mat.T).todense().any()
    else:
        return not (mat!=mat.T).any()
    
def adj_norm(adj, issym=True):
    dev=0
    if torch.is_tensor(adj):
        dev = adj.device
        adj = to_dense(adj).cpu().numpy()
  
    adj = sp.csc_matrix(adj)
    if issym:
        rowsum = np.array(adj.sum(1))
        colsum = np.array(adj.sum(0))
        rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
        coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
        adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv)
    else:
        rowsum = np.array(adj.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.nan_to_num(np.power(rowsum, -1)).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj)
        
    return sparse_to_tensor(adj_normalized).cuda(dev)

def maxminnorm(df):
    min=df.min(axis=0)
    max=df.max(axis=0)
    return (df-min)/(max-min)


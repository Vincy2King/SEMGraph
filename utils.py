import numpy as np
import pickle as pkl
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import sys
import random
import re
from tqdm import tqdm
# import sparse


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_information(information):
    max_length = max([len(f) for f in information])
    list_1 = []
    for i in tqdm(range(information.shape[0])):
        list_2=[]
        info = np.array(information[i])
        pad = max_length - info.shape[0]  # padding for each epoch
        info = np.pad(info, (pad, 0), mode='constant')
        for k in range(len(info)):
            list_3 = []
            # each_weight=np.array(list_3.append(weight[k]/total))
            list_3.append(info[k])
            list_2.append(list_3)
        list_1.append(np.array(list_2))
        # print('weight:',weight.shape)
        information[i] = info
    # print('list_1:',np.array(list_1).shape)
    return np.array(list_1)

def load_weight(weights):
    # weight_list=[]
    # for each_list in dataset_weight:
    #     # each_list=np.array(each_list)
    #     print('type:',each_list.shape)
    #     weight_list.append(each_list)
    #     # print('each_',each_list)
    # print('np.array(weight_list).shape:',np.array(weight_list).shape)
    # return np.array(weight_list)

    max_length = max([len(f) for f in weights])
    list_1=[]
    print(max_length)
    for i in tqdm(range(weights.shape[0])):
        weight = np.array(weights[i])
        pad = max_length - weight.shape[0]  # padding for each epoch
        weight = np.pad(weight, (pad,0), mode='constant')
        # print(len(weight))
        list_2=[]
        total=0
        minn=999
        maxx=-999
        for k in range(len(weight)):
            if weight[k]>=maxx:
                maxx=weight[k]
            if weight[k]<=minn:
                minn=weight[k]
        # 归一化
        for k in range(len(weight)):
            weight[k]=(weight[k]-minn)/(maxx-minn)
            total+=weight[k]
        for k in range(len(weight)):
            list_3=[]
            # each_weight=np.array(list_3.append(weight[k]/total))
            list_3.append(weight[k] / total)
            list_2.append(list_3)
        list_1.append(np.array(list_2))
        # print('weight:',weight.shape)

        weights[i] = weight
    # print('list_1:',np.array(list_1).shape)

    return np.array(list_1)
    # return np.array(list(weights))

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors and adjacency matrix of the training instances as list;
    ind.dataset_str.tx => the feature vectors and adjacency matrix of the test instances as list;
    ind.dataset_str.allx => the feature vectors and adjacency matrix of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as list;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x_adj', 'x_embed','x_FFD_weight','x_TRT_weight','x_FFD1_weight','x_TRT1_weight','x_FFD2_weight','x_TRT2_weight', 'y',
             'tx_adj', 'tx_embed','tx_FFD_weight','tx_TRT_weight','tx_FFD1_weight','tx_TRT1_weight','tx_FFD2_weight','tx_TRT2_weight', 'ty',
             'allx_adj', 'allx_embed','allx_FFD_weight','allx_TRT_weight','allx_FFD1_weight','allx_TRT1_weight','allx_FFD2_weight','allx_TRT2_weight', 'ally']
    objects = []
    for i in range(len(names)):
        with open("data/{}/ind.{}.{}".format(dataset_str,dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
            try:
                objects.append(pkl.load(f))
            except EOFError:
                print('None')
    x_adj, x_embed,x_FFD_weight,x_TRT_weight,x_FFD1_weight,x_TRT1_weight,x_FFD2_weight,x_TRT2_weight, y, \
    tx_adj, tx_embed, tx_FFD_weight,tx_TRT_weight,tx_FFD1_weight,tx_TRT1_weight,tx_FFD2_weight,tx_TRT2_weight,ty, \
    allx_adj, allx_embed,allx_FFD_weight,allx_TRT_weight,allx_FFD1_weight,allx_TRT1_weight,allx_FFD2_weight,allx_TRT2_weight, ally = tuple(objects)
    # train_idx_ori = parse_index_file("data/{}.train.index".format(dataset_str))
    # train_size = len(train_idx_ori)

    train_adj = []
    train_embed = []
    train_FFD_weight = []
    train_TRT_weight = []
    train_FFD1_weight = []
    train_TRT1_weight = []
    train_FFD2_weight = []
    train_TRT2_weight = []

    val_adj = []
    val_embed = []
    val_FFD_weight = []
    val_TRT_weight = []
    val_FFD1_weight = []
    val_TRT1_weight = []
    val_FFD2_weight = []
    val_TRT2_weight = []

    test_adj = []
    test_embed = []
    test_FFD_weight = []
    test_TRT_weight = []
    test_FFD1_weight = []
    test_TRT1_weight = []
    test_FFD2_weight = []
    test_TRT2_weight = []
    # print(x_embed)
    for i in range(len(y)):
        adj = x_adj[i].toarray()
        embed = np.array(x_embed[i])
        FFD_weight=np.array(x_FFD_weight[i])
        TRT_weight = np.array(x_TRT_weight[i])
        FFD1_weight = np.array(x_FFD1_weight[i])
        TRT1_weight = np.array(x_TRT1_weight[i])
        FFD2_weight = np.array(x_FFD2_weight[i])
        TRT2_weight = np.array(x_TRT2_weight[i])
        train_adj.append(adj)
        train_embed.append(embed)
        train_FFD_weight.append(FFD_weight)
        train_TRT_weight.append(TRT_weight)
        train_FFD1_weight.append(FFD1_weight)
        train_TRT1_weight.append(TRT1_weight)
        train_FFD2_weight.append(FFD2_weight)
        train_TRT2_weight.append(TRT2_weight)

    for i in range(len(y), len(ally)): #train_size):
        adj = allx_adj[i].toarray()
        embed = np.array(allx_embed[i])
        FFD_weight = np.array(allx_FFD_weight[i])
        TRT_weight = np.array(allx_TRT_weight[i])
        FFD1_weight = np.array(allx_FFD1_weight[i])
        TRT1_weight = np.array(allx_TRT1_weight[i])
        FFD2_weight = np.array(allx_FFD2_weight[i])
        TRT2_weight = np.array(allx_TRT2_weight[i])
        val_adj.append(adj)
        val_embed.append(embed)
        val_FFD_weight.append(FFD_weight)
        val_TRT_weight.append(TRT_weight)
        val_FFD1_weight.append(FFD1_weight)
        val_TRT1_weight.append(TRT1_weight)
        val_FFD2_weight.append(FFD2_weight)
        val_TRT2_weight.append(TRT2_weight)

    for i in range(len(ty)):
        adj = tx_adj[i].toarray()
        embed = np.array(tx_embed[i])
        FFD_weight = np.array(tx_FFD_weight[i])
        TRT_weight = np.array(tx_TRT_weight[i])
        FFD1_weight = np.array(tx_FFD1_weight[i])
        TRT1_weight = np.array(tx_TRT1_weight[i])
        FFD2_weight = np.array(tx_FFD2_weight[i])
        TRT2_weight = np.array(tx_TRT2_weight[i])
        test_adj.append(adj)
        test_embed.append(embed)
        test_FFD_weight.append(FFD_weight)
        test_TRT_weight.append(TRT_weight)
        test_FFD1_weight.append(FFD1_weight)
        test_TRT1_weight.append(TRT1_weight)
        test_FFD2_weight.append(FFD2_weight)
        test_TRT2_weight.append(TRT2_weight)

    train_adj = np.array(train_adj)
    val_adj = np.array(val_adj)
    test_adj = np.array(test_adj)
    train_embed = np.array(train_embed)
    val_embed = np.array(val_embed)
    test_embed = np.array(test_embed)
    print('y_shape:',y.shape)
    print('y:',y)
    train_y = np.array(y)
    val_y = np.array(ally[len(y):len(ally)]) # train_size])
    test_y = np.array(ty)
    
    train_FFD_weight = np.array(train_FFD_weight)
    train_TRT_weight = np.array(train_TRT_weight)
    train_FFD1_weight = np.array(train_FFD1_weight)
    train_TRT1_weight = np.array(train_TRT1_weight)
    train_FFD2_weight = np.array(train_FFD2_weight)
    train_TRT2_weight = np.array(train_TRT2_weight)

    val_FFD_weight = np.array(val_FFD_weight)
    val_TRT_weight = np.array(val_TRT_weight)
    val_FFD1_weight = np.array(val_FFD1_weight)
    val_TRT1_weight = np.array(val_TRT1_weight)
    val_FFD2_weight = np.array(val_FFD2_weight)
    val_TRT2_weight = np.array(val_TRT2_weight)

    test_FFD_weight = np.array(test_FFD_weight)
    test_TRT_weight = np.array(test_TRT_weight)
    test_FFD1_weight = np.array(test_FFD1_weight)
    test_TRT1_weight = np.array(test_TRT1_weight)
    test_FFD2_weight = np.array(test_FFD2_weight)
    test_TRT2_weight = np.array(test_TRT2_weight)

    return train_adj, train_embed,train_FFD_weight, train_TRT_weight,train_FFD1_weight, train_TRT1_weight,train_FFD2_weight, train_TRT2_weight,train_y,\
           val_adj, val_embed, val_FFD_weight,val_TRT_weight,val_FFD1_weight,val_TRT1_weight,val_FFD2_weight,val_TRT2_weight,val_y, \
           test_adj, test_embed, test_FFD_weight,test_TRT_weight,test_FFD1_weight,test_TRT1_weight,test_FFD2_weight,test_TRT2_weight,test_y


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def coo_to_tuple(sparse_coo):
    return (sparse_coo.coords.T, sparse_coo.data, sparse_coo.shape)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    max_length = max([len(f) for f in features])
    print('max_length:',max_length)
    for i in tqdm(range(features.shape[0])):
        feature = np.array(features[i])
        pad = max_length - feature.shape[0] # padding for each epoch
        feature = np.pad(feature, ((0,pad),(0,0)), mode='constant')
        features[i] = feature
        # print('len(feature):',feature.shape)
    return np.array(list(features))


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    max_length = max([a.shape[0] for a in adj])
    mask = np.zeros((adj.shape[0], max_length, 1)) # mask for padding

    for i in tqdm(range(adj.shape[0])):
        adj_normalized = normalize_adj(adj[i]) # no self-loop
        pad = max_length - adj_normalized.shape[0] # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
        mask[i,:adj[i].shape[0],:] = 1.
        adj[i] = adj_normalized

    return np.array(list(adj)), mask # coo_to_tuple(sparse.COO(np.array(list(adj)))), mask


def construct_feed_dict(FFD_info,TRT_info,FFD_weights,TRT_weights,features, support, mask, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['FFD_info']: FFD_info})
    feed_dict.update({placeholders['TRT_info']: TRT_info})
    feed_dict.update({placeholders['FFD_weight']: FFD_weights})
    feed_dict.update({placeholders['TRT_weight']: TRT_weights})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['mask']: mask})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

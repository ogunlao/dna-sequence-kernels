import pandas as pd
import numpy as np
import os
import warnings
import kernels as km
import pickle as pkl
import datetime
from tqdm import tqdm

path = './data'

def get_train(path=path):
    """
    Load training data set and replace 0 by -1 in target.
    :param path: String, path of data
    :return:
        - X: pd.DataFrame, sequences
        - y: pd.DataFrame, labels (0/1)
    """
    x_train_path = os.path.join(path, 'Xtr.csv')
    y_train_path = os.path.join(path, 'Ytr.csv')

    X, y = pd.read_csv(x_train_path), pd.read_csv(y_train_path)
    y['Bound'] = y['Bound'].replace(0, -1)
    return X, y

def get_test(path=path):
    """
    Load testing data set from path
    :param path: String, path of data
    :return:
        - X: pd.DataFrame, sequences
    """
    x_test_path = path+'/Xte.csv'
    X = pd.read_csv(x_test_path)
    return X

def train_test_splitter(path=path, s=0.8):
    """
    Split training data into training and testing set, with split ratio s=0.8
    Training and testing data sets are balanced w.r.t to the target y.
    :param s: float between 0 and 1, split proportion
    :return:
        - X_train: pd.DataFrame, training sequences
        - X_val: pd.DataFrame, validation sequences
        - y_train: pd.DataFrame, training labels
        - y_val: pd.DataFrame, validation labels
        - X_test: pd.DataFrame, testing sequences
    """
    X, y = get_train(path=path)

    idx_0, idx_1 = np.where(y.loc[:, "Bound"] == -1)[0], np.where(y.loc[:, "Bound"] == 1)[0]
    n0, n1 = len(idx_0), len(idx_1)
    idx_tr0, idx_tr1 = idx_0[:int(s * n0)+1], idx_1[:int(s * n1)+1]
    idx_te0, idx_te1 = list(set(idx_0) - set(idx_tr0)), list(set(idx_1) - set(idx_tr1))
    idx_tr, idx_te = np.concatenate((idx_tr0, idx_tr1)), np.concatenate((idx_te0, idx_te1))
    
    X_train = X.iloc[idx_tr, :].reset_index(drop=True)
    y_train = y.iloc[idx_tr, :].reset_index(drop=True)

    X_val = X.iloc[idx_te, :].reset_index(drop=True)
    y_val = y.iloc[idx_te, :].reset_index(drop=True)

    X_test = get_test(path=path)
    return X_train, y_train, X_val, y_val, X_test

def get_training_datas(method, all=True, replace=False):
    """
    Construct training, testing data, and kernels.
    :param: method: string, method used for computing kernels
    :param: replace: Boolean, whether or not replace the existing files in the repo
    :return:
        - X_train: pd.DataFrame, training sequences
        - X_val: pd.DataFrame, validation sequences
        - y_train: pd.DataFrame, training labels
        - y_val: pd.DataFrame, validation labels
        - X_test: pd.DataFrame, testing sequences
        - K: np.array, kernel
        - ID: np.array, Ids
    """
    file = 'training_data_'+method+'.pkl'
    if not all:
        X_train, y_train, X_val, y_val, X_test = train_test_split()
        X_test.loc[:, 'Id'] = -(X_test.loc[:, 'Id'] + 1)
        X = pd.concat((X_train, X_val, X_test), axis=0)
        ID = X.loc[:, 'Id']
        
    else:
        if trainInRepo(file) and not replace:
            X_train, y_train, X_val, y_val, X_test, K, ID = pkl.load(open(os.path.join(path, file), 'rb'))
            
        else:
            X_train, y_train, X_val, y_val, X_test = train_test_split()
            X_test.loc[:, 'Id'] = -(X_test.loc[:, 'Id']+1)
            X = pd.concat((X_train, X_val, X_test), axis=0)
            ID = np.array(X.loc[:, 'Id'])
            K = km.select_method(X, method)
            file = 'training_data_'+method+'.pkl'
            pkl.dump([X_train, y_train, X_val, y_val, X_test, K, ID], open(os.path.join(path, file), 'wb'))
    return X_train, y_train, X_val, y_val, X_test, K, ID


def get_data(method, all=True, replace=False):
    """
    Construct training, testing data, and kernels.
    :param: method: string, method used for computing kernels
    :param: replace: Boolean, whether or not replace the existing files in the repo
    :return:
        - X_train: pd.DataFrame, training sequences
        - X_val: pd.DataFrame, validation sequences
        - y_train: pd.DataFrame, training labels
        - y_val: pd.DataFrame, validation labels
        - X_test: pd.DataFrame, testing sequences
        - K: np.array, kernel
        - ID: np.array, Ids
    """
    file = 'training_data_'+method+'.pkl'
    if not all:
        X_train, y_train, X_val, y_val, X_test = train_test_splitter(path=path, s=0.8)
        X_test.loc[:, 'Id'] = -(X_test.loc[:, 'Id'] + 1)
        X = pd.concat((X_train, X_val, X_test), axis=0)
        ID = X.loc[:, 'Id']
        
    else:
        if file in os.listdir(path) and not replace:
            X_train, y_train, X_val, y_val, X_test, K, ID = pkl.load(open(os.path.join(path, file), 'rb'))

        else:
            
            X_train, y_train, X_val, y_val, X_test = train_test_splitter(path=path, s=0.8)
            X_test.loc[:, 'Id'] = -(X_test.loc[:, 'Id']+1)
            X = pd.concat((X_train, X_val, X_test), axis=0)
            ID = np.array(X.loc[:, 'Id'])
            K = km.select_method(X, method)
            file = 'training_data_'+method+'.pkl'
            pkl.dump([X_train, y_train, X_val, y_val, X_test, K, ID], open(os.path.join(path, file), 'wb'))
    return X_train, y_train, X_val, y_val, X_test, K, ID

def load_kernel(method):
    """
    Load kernel
    :param method: string, kernel method
    :return: np.array, kernel
    """
    _, _, _, _, _, K, _ = get_data(method=method, replace=False)
    return K

def get_saved_data(methods):
    """
    Return all the necessary data for training and running the experiments.
    :param methods: list of strings, kernel methods
    :return: list:
            - X_train, y_train, X_val, y_val, X_test
            - kernels (list of np.array kernels)
            - IDs (np.array)
    """
    print('Loading data...')
    X_train, y_train, X_val, y_val, X_test, K, ID = get_data(method=methods[0], replace=False)
    data = [X_train, y_train, X_val, y_val, X_test]
    kernels = []
    for k, m in enumerate(methods):
        print('Kernel '+str(k+1)+'...')
        kernels.append(load_kernel(m))
    if len(kernels) == 1:
        kernels = kernels[0]
    return data, kernels, ID

def reformat_data(data, kernels, ID):
    X_train, y_train, X_val, y_val, X_test = data
    ID_ = np.concatenate(
        (np.array(X_train.loc[:, 'Id']), np.array(X_val.loc[:, 'Id']), np.array(X_test.loc[:, 'Id'])))
    idx = np.array([np.where(ID == ID_[i])[0] for i in range(len(ID_))]).squeeze()
    kernels_ = []
    for K in tqdm(kernels):
        kernels_.append(K[idx][:, idx])

    ID_ = np.arange(ID_.shape[0])
    X_train.Id = ID_[:X_train.shape[0]]
    X_val.Id = ID_[X_train.shape[0]:(X_train.shape[0] + X_val.shape[0])]
    X_test.Id = ID_[(X_train.shape[0] + X_val.shape[0]):(
            X_train.shape[0] + X_val.shape[0] + X_test.shape[0])]
    y_train.Id = ID_[:y_train.shape[0]]
    y_val.Id = ID_[X_train.shape[0]:(X_train.shape[0] + X_val.shape[0])]
    return X_train, y_train, X_val, y_val, X_test, kernels_, ID_

def save_predictions(model, X_test):
    """
    Compute and save predictions to folder
    :param model: trained model to be used for prediction
    :param X_tests: list, list of testing pd.DataFrames
    :return: y_df, dataframe containing predictions
    """
    len_of_pred = len(X_test)

    y_pred = model.predict(X_test).astype(int)
    y_df = pd.DataFrame({'Id': X_test.Id, 'Bound': y_pred})
    y_df.Id = np.arange(len_of_pred)
    y_df.Bound = y_df.Bound.replace(-1, 0)
    y_df.to_csv('Yte.csv', index=False)

    return y_df
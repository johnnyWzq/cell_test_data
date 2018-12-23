#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:58:35 2018

@author: wuzhiqiang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 19:33:14 2018

@author: zhiqiangwu
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_validate
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn import tree
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def select_feature(data_x, data_y, feature_num=100, method='f_regression'):
    features_chosen = data_x.columns
    feature_num = min(len(features_chosen), feature_num)
    
     # standardize
    """
    min_max = pd.DataFrame([data_x.min(), data_x.max()])
    scaler = MinMaxScaler(feature_range=(-1,1))
    print(scaler.fit(data_x.values))
    data_x = pd.DataFrame(data=scaler.transform(data_x.values), columns=data_x.columns)
    """
    min_max = None
    #data_x = pd.DataFrame(data=preprocessing.Scaler(data_x.values, axis=0), columns=data_x.columns)
    #根据特征工程的方法选择特征参数数量
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_regression
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_regression
    
    if method == 'f_regression' or method == 'mutual_info_regression':
        if method == 'f_regression':
            select_model = SelectKBest(f_regression, k=feature_num)
        else:
            select_model = SelectKBest(mutual_info_regression, k=feature_num)
        select_model.fit(data_x.values, data_y.values.ravel())
        feature_mask = select_model.get_support(indices=True)
        feature_chosen = data_x.columns[feature_mask]
        print('feature_chosen: ', feature_chosen)
        data_x = data_x[feature_chosen]
    elif method == 'PCA':
        pca_model = PCA(n_components=feature_num)
        data_x_pc = pca_model.fit(data_x.values).transform(data_x.values)
        data_x = pd.DataFrame(data=data_x_pc,
                      columns=['PCA_' + str(i) for i in range(feature_num)])
    else:
        raise Exception('In select_feature(): invalid parameter method.')
    
    return data_x, min_max

def evaluate(model, X, trueY):
    """
    util function
    :param model: model to be evaluated
    :param X: X
    :param trueY: true Y
    :return: model's information
    """
    predY = model.predict(X)
    scores = {}
    scores['EVS'] = explained_variance_score(trueY, predY)
    scores['MAE'] = mean_absolute_error(trueY, predY)
    scores['MSE'] = mean_squared_error(trueY, predY)
    scores['R2'] = r2_score(trueY, predY)
    
    return scores

def eva_merge(eva1, eva2, index):
    eva1['type'] = 'train'
    eva2['type'] = 'test'
    df1 = pd.DataFrame(eva1, index=[index])
    df2 = pd.DataFrame(eva2, index=[index])
    df = pd.concat([df1, df2])
    return df

def test_model(model, x_train, x_val, y_train, y_val):
    """
    util function: train and test a model given training-set and testing-set
    :param model: model untrained
    :param x_train: training-set's X
    :param x_val: testing-set's X
    :param y_train: training-set's Y
    :param y_val: testing-set's Y
    :return:
    """
    print(model)
    model.fit(x_train, y_train)

    eva1 = evaluate(model, x_train, y_train)
    eva2 = evaluate(model, x_val, y_val)
    print('evaluation result of training set: \n', eva1)
    print('evaluation result of validation set: \n', eva2)
    print('----------------------------------------')
    get_model_name = lambda x:x[0:x.find('(')]
    res = {}
    res['train'] = pd.DataFrame({'train:true_y': y_train, 'train:pred_y': model.predict(x_train)})
    res['test'] = pd.DataFrame({'test:true_y': y_val, 'test:pred_y': model.predict(x_val)})
    res['eva'] = eva_merge(eva1, eva2, get_model_name(str(model)))
    
    return res

def cv_model(model, allX, allY, cv_num=10):
    """
    util function: cross-validate a model given all data
    :param model: model to be cross-validated
    :param allX: all data's X
    :param allY: all data's Y
    :param cv_num: cross-validation's fold number
    :return:
    """
    print('----------------------------------------')
    print(model)
    print('cv_num=%s' %(cv_num))
    scoring = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
    scores = cross_validate(model, allX, allY, scoring=scoring, cv=cv_num,
                            return_train_score=True)
    for i in scores:
        print("%s: %0.4f (+/- %0.4f)" % (i, scores[i].mean(), scores[i].std() * 2))
    return scores
        
def save_model(model, feature_name, pkl_dir, depth=None):
    get_model_name = lambda x:x[0:x.find('(')]
    model_name = get_model_name(str(model))
    joblib.dump(model, os.path.join(pkl_dir, model_name+'.pkl')) 

    if depth == None:
        depth = 'x'
    if model_name == 'DecisionTreeRegressor':
        tree.export_graphviz(model, os.path.join(pkl_dir, 'dt_depth%s.dot'%str(depth)),
                             feature_names=feature_name, max_depth=3)
    elif model_name == 'GradientBoostingRegressor':
        tree.export_graphviz(model.estimators_.ravel()[0], os.path.join(pkl_dir, 'gbdt.dot'),
                             feature_names=feature_name)
    elif model_name == 'RandomForestRegressor':
        tree.export_graphviz(model.estimators_[0], os.path.join(pkl_dir, 'rf_depth%s.dot'%str(depth)),
                             feature_names=feature_name, max_depth=3)
        
def load_model(model_name):
    model = joblib.load(model_name)
    return model

def valid_model(model, data_x, data_y, feature_method, feature_num=100):
    data_x, min_max = select_feature(data_x, data_y, method=feature_method, feature_num=feature_num)#min_max no uses
    data_x = data_x[0:-2]
    data_y = data_y[0:-2]
    # start building model
    np_x = np.nan_to_num(data_x.values)
    np_y = np.nan_to_num(data_y.values)
    print('train_set.shape=%s, test_set.shape=%s' %(np_x.shape, np_y.shape))
    eva2 = evaluate(model, np_x, np_y)
    print('evaluation result of validation set: \n', eva2)
    print('----------------------------------------')
    res = pd.DataFrame({'test:true_y': np_y, 'test:pred_y': model.predict(np_x)})
    return res

def add_min_max(data_x, data_y, file_name):
    min_max = pd.read_csv(file_name, encoding='gb18030')
    data_x = data_x.append(min_max)
    data_y = data_y.append(pd.Series([data_y.min(), data_y.max()]))
    return data_x, data_y
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:43:33 2018

@author: wuzhiqiang
"""

print(__doc__)

import os
import pandas as pd
import numpy as np
import utils as ut

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
    
def calc_score(data):
    """
    """
    RATE_CAPACITY = data['c'].max()
    
    data['score'] = data['c'] / RATE_CAPACITY

    return data

def calc_feature_data(file_name):
    """
    """
    data = calc_feature(file_name)
    data = calc_score(data)
    
    data_x = data[[i for i in data.columns if 'feature_' in i]]
    data_y = data['score']
    
    return data_x, data_y

def transfer_feature(data):
    """
    curr-c_rate,voltage-v_rate,T-T_refer
    """
    C_RATE = 37
    V_RATE = 3.7
    T_REFER = 20
    data['current_mean'] = data['current_mean'] / C_RATE
    data['current_min'] = data['current_min'] / C_RATE
    data['current_max'] = data['current_max'] / C_RATE
    data['current_median'] = data['current_median'] / C_RATE
    data['current_std'] = data['current_std'] / C_RATE
    
    data['voltage_mean'] = data['voltage_mean'] / V_RATE
    data['voltage_min'] = data['voltage_min'] / V_RATE
    data['voltage_max'] = data['voltage_max'] / V_RATE
    data['voltage_median'] = data['voltage_median'] / V_RATE
    data['voltage_std'] = data['voltage_std'] / V_RATE
    
    data['temperature_mean'] = data['temperature_mean'] / T_REFER
    data['temperature_min'] = data['temperature_min'] / T_REFER
    data['temperature_max'] = data['temperature_max'] / T_REFER
    data['temperature_median'] = data['temperature_median'] / T_REFER
    data['temperature_std'] = data['temperature_std'] / T_REFER
    
    data['dqdv_mean'] = data['dqdv_mean'] / C_RATE / V_RATE
    data['dqdv_min'] = data['dqdv_min'] / C_RATE / V_RATE
    data['dqdv_max'] = data['dqdv_max'] / C_RATE / V_RATE
    data['dqdv_median'] = data['dqdv_median'] / C_RATE / V_RATE
    data['dqdv_std'] = data['dqdv_std'] / C_RATE / V_RATE
    
    return data

def calc_feature(file_name):
    """
    """
    print('---------------------------------------')
    print(file_name)
    data = pd.read_csv(file_name, encoding='gb18030')
    print('data shape: ' + data.shape.__str__())
    
     #transfer the feature
    data = transfer_feature(data)
    #clearing
    invaid_thresh = data.shape[0] // 4 * 3
    data = data.dropna(axis='columns', thresh=invaid_thresh)
    print(data.shape)
    data = data.T[~data.isin([-np.inf, np.inf]).all().values]#删除所有行均为-inf,inf的列
    data = data.T
    print(data.shape)
    data = data.replace(np.inf, np.nan)
    data = data.replace(-np.inf, np.nan)
    data = data.dropna(axis='columns', thresh=invaid_thresh)
    print(data.shape)
    data = data.fillna(data.mean())
    
    col_names = []
    for i in data.columns:
        for j in ['voltage_', 'current_', 'temperatrue', 'dqdv_']:
            if j in i:
                col_names.append(i)
                break
    for i in col_names:
        tmp = data[i]
        data['feature_' + i] = tmp
        
    return data

def select_feature(data_x, data_y, feature_num=40, method='f_regression'):
    features_chosen = data_x.columns
    feature_num = min(len(features_chosen), feature_num)
    
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
    
    return data_x

def build_model(data_x, data_y, split_mode='test',
                feature_method='f_regression', feature_num=40):

     # standardize
    data_x = pd.DataFrame(data=preprocessing.scale(data_x.values, axis=0), columns=data_x.columns)

    # select features
    # feature_num: integer that >=0
    # method: ['f_regression', 'mutual_info_regression', 'pca']
    data_x = select_feature(data_x, data_y, method=feature_method, feature_num=feature_num)
    
    # start building model
    np_x = np.nan_to_num(data_x.values)
    np_y = np.nan_to_num(data_y.values)
    print('train_set.shape=%s, test_set.shape=%s' %(np_x.shape, np_y.shape))

    pkl_dir = os.path.join(os.path.abspath('.'), 'pkl')
    res = {}
    if split_mode == 'test':
        x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.2,
                                                          shuffle=False)
        model = LinearRegression()
        res['lr'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir)
        model = DecisionTreeRegressor()
        res['dt'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir, depth=3)
        model = RandomForestRegressor(min_samples_split=2)
        res['rf'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir, depth=3)
        model = GradientBoostingRegressor()
        res['gbdt'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir)
    elif split_mode == 'cv':
        model = LinearRegression()
        res['lr'] = ut.cv_model(model, np_x, np_y)
        model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=3)
        res['dt'] = ut.cv_model(model, np_x, np_y)
        model = RandomForestRegressor()
        res['rf'] = ut.cv_model(model, np_x, np_y)
        model = GradientBoostingRegressor()
        res['gbdt'] = ut.cv_model(model, np_x, np_y)
    else:
        print('parameter mode=%s unresolved' % (model))
        
    return res
    
def main():
    state = 'charge'
    file_dir = os.path.join(os.path.abspath('.'), 'data')
    file_name = os.path.join(os.path.join(file_dir, r'processed_%s_data.csv'%state))
    data_x, data_y = calc_feature_data(file_name)
    #feature_data.to_excel(os.path.join(file_dir, 'feature_data.xlsx'))
      
    mode = 'test'
    res = build_model(data_x, data_y, split_mode=mode, feature_method='f_regression')
    if mode == 'test':
        d = {'lr':'线性回归(LR)', 'dt':'决策树回归', 'rf':'随机森林', 'gbdt':'GBDT',
            'eva':'评估结果'}
        writer = pd.ExcelWriter(os.path.join(os.path.join(os.path.abspath('.'), 'result'), 'result.xlsx'))
        eva = pd.DataFrame()
        for s in res:
            res[s]['train'].to_excel(writer, d[s])
            res[s]['test'].to_excel(writer, d[s], startcol=3)
            eva = eva.append(res[s]['eva'])
        eva = eva[['type', 'EVS', 'MAE', 'MSE', 'R2']]
        eva.to_excel(writer, d['eva'])
    
if __name__ == '__main__':
    main()
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
import read_data as rd

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
    RATE_CAPACITY = 38.0#data['c'].max()
    
    data['score'] = data['c'] / RATE_CAPACITY

    return data

def calc_feature_data(file_dir, file_name, data=None):
    """
    """
    data = calc_feature(file_dir, file_name, data)
    data = calc_score(data)
    data = data[data['score'] > 0.1]
    
    print('getting the feature...')
    data_x = data[[i for i in data.columns if 'feature_' in i]]
    #将电流特征去掉
    data_x = data_x.drop([i for i in data.columns if '_current_' in i], axis=1)
    data_y = data['score']
    
    return data_x, data_y

def transfer_feature(data):
    """
    curr-c_rate,voltage-v_rate,T-T_refer
    """
    print('normalizating the data... ')
    C_RATE = 37
    V_RATE = 3.7
    T_REFER = 20
    data['current_mean'] = data['current_mean'] / C_RATE
    data['current_min'] = data['current_min'] / C_RATE
    data['current_max'] = data['current_max'] / C_RATE
    data['current_median'] = data['current_median'] / C_RATE
    data['current_std'] = data['current_std'] / C_RATE
    
    data['current_diff_mean'] = data['current_diff_mean'] / C_RATE
    data['current_diff_min'] = data['current_diff_min'] / C_RATE
    data['current_diff_max'] = data['current_diff_max'] / C_RATE
    data['current_diff_median'] = data['current_diff_median'] / C_RATE
    data['current_diff_std'] = data['current_diff_std'] / C_RATE
    
    data['current_diff2_mean'] = data['current_diff2_mean'] / C_RATE
    data['current_diff2_min'] = data['current_diff2_min'] / C_RATE
    data['current_diff2_max'] = data['current_diff2_max'] / C_RATE
    data['current_diff2_median'] = data['current_diff2_median'] / C_RATE
    data['current_diff2_std'] = data['current_diff2_std'] / C_RATE
    
    data['voltage_mean'] = data['voltage_mean'] / V_RATE
    data['voltage_min'] = data['voltage_min'] / V_RATE
    data['voltage_max'] = data['voltage_max'] / V_RATE
    data['voltage_median'] = data['voltage_median'] / V_RATE
    data['voltage_std'] = data['voltage_std'] / V_RATE
    
    data['voltage_diff_mean'] = data['voltage_diff_mean'] / V_RATE
    data['voltage_diff_min'] = data['voltage_diff_min'] / V_RATE
    data['voltage_diff_max'] = data['voltage_diff_max'] / V_RATE
    data['voltage_diff_median'] = data['voltage_diff_median'] / V_RATE
    data['voltage_diff_std'] = data['voltage_diff_std'] / V_RATE
    
    data['voltage_diff2_mean'] = data['voltage_diff2_mean'] / V_RATE
    data['voltage_diff2_min'] = data['voltage_diff2_min'] / V_RATE
    data['voltage_diff2_max'] = data['voltage_diff2_max'] / V_RATE
    data['voltage_diff2_median'] = data['voltage_diff2_median'] / V_RATE
    data['voltage_diff2_std'] = data['voltage_diff2_std'] / V_RATE
    
    data['temperature_mean'] = data['temperature_mean'] / T_REFER
    data['temperature_min'] = data['temperature_min'] / T_REFER
    data['temperature_max'] = data['temperature_max'] / T_REFER
    data['temperature_median'] = data['temperature_median'] / T_REFER
    data['temperature_std'] = data['temperature_std'] / T_REFER
    
    data['dqdv_mean'] = data['dqdv_mean'] / C_RATE * V_RATE
    data['dqdv_min'] = data['dqdv_min'] / C_RATE * V_RATE
    data['dqdv_max'] = data['dqdv_max'] / C_RATE * V_RATE
    data['dqdv_median'] = data['dqdv_median'] / C_RATE * V_RATE
    data['dqdv_std'] = data['dqdv_std'] / C_RATE * V_RATE
    
    data['dqdv_diff_mean'] = data['dqdv_diff_mean'] / C_RATE * V_RATE
    data['dqdv_diff_min'] = data['dqdv_diff_min'] / C_RATE * V_RATE
    data['dqdv_diff_max'] = data['dqdv_diff_max'] / C_RATE * V_RATE
    data['dqdv_diff_median'] = data['dqdv_diff_median'] / C_RATE * V_RATE
    data['dqdv_diff_std'] = data['dqdv_diff_std'] / C_RATE * V_RATE
    
    data['dqdv_diff2_mean'] = data['dqdv_diff2_mean'] / C_RATE * V_RATE
    data['dqdv_diff2_min'] = data['dqdv_diff2_min'] / C_RATE * V_RATE
    data['dqdv_diff2_max'] = data['dqdv_diff2_max'] / C_RATE * V_RATE
    data['dqdv_diff2_median'] = data['dqdv_diff2_median'] / C_RATE * V_RATE
    data['dqdv_diff2_std'] = data['dqdv_diff2_std'] / C_RATE * V_RATE
    
    return data

def calc_feature(file_dir, file_name, data=None):
    """
    """
    print('---------------------------------------')
    if data is None:
        print(file_name)
        file = os.path.join(os.path.join(file_dir, file_name))
        data = pd.read_csv(file, encoding='gb18030')
    print('data shape: ' + data.shape.__str__())
    
     #transfer the feature
    data = transfer_feature(data)
    #clearing
    invaid_thresh = data.shape[0]# // 4 * 3
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

def build_model(data_x, data_y, split_mode='test',
                feature_method='f_regression', feature_num=100, pkl_dir='pkl'):
    # select features
    # feature_num: integer that >=0
    # method: ['f_regression', 'mutual_info_regression', 'pca']
    data_x, min_max = ut.select_feature(data_x, data_y, method=feature_method, feature_num=feature_num)
    if min_max is not None:
        min_max.to_csv(os.path.join(pkl_dir, 'min_max.csv'), index=False, index_label=False, encoding='gb18030')
    # start building model
    np_x = np.nan_to_num(data_x.values)
    np_y = np.nan_to_num(data_y.values)
    print('train_set.shape=%s, test_set.shape=%s' %(np_x.shape, np_y.shape))

    res = {}
    if split_mode == 'test':
        x_train, x_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.2,
                                                          shuffle=True)
        model = LinearRegression()
        res['lr'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir)
        model = DecisionTreeRegressor()
        res['dt'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir, depth=3)
        model = RandomForestRegressor()
        res['rf'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir, depth=3)
        model = GradientBoostingRegressor()
        res['gbdt'] = ut.test_model(model, x_train, x_val, y_train, y_val)
        ut.save_model(model, data_x.columns, pkl_dir)
    elif split_mode == 'cv':
        model = LinearRegression()
        res['lr'] = ut.cv_model(model, np_x, np_y)
        model = DecisionTreeRegressor()
        res['dt'] = ut.cv_model(model, np_x, np_y)
        model = RandomForestRegressor()
        res['rf'] = ut.cv_model(model, np_x, np_y)
        model = GradientBoostingRegressor()
        res['gbdt'] = ut.cv_model(model, np_x, np_y)
    else:
        print('parameter mode=%s unresolved' % (model))
        
    return res
    
def main():
    state = 'discharge'
    file_dir = os.path.join(os.path.abspath('.'), 'data')
    file_name = r'scale_processed_%s_data.csv'%state
    pkl_dir = os.path.join(os.path.abspath('.'), '%s_g_pkl'%state)
    data_x, data_y = calc_feature_data(file_dir, file_name)

    rd.save_data_csv(data_x.head(), 'columns_feature_'+file_name[:-4], pkl_dir)
    mode = 'test'
    res = build_model(data_x, data_y, split_mode=mode, feature_method='f_regression', pkl_dir=pkl_dir)
    if mode == 'test':
        d = {'lr':'线性回归(LR)', 'dt':'决策树回归', 'rf':'随机森林', 'gbdt':'GBDT',
            'eva':'评估结果'}
        writer = pd.ExcelWriter(os.path.join(os.path.join(os.path.abspath('.'), 'g_result'),
                                             '%s_result.xlsx'%state))
        eva = pd.DataFrame()
        for s in res:
            res[s]['train'].to_excel(writer, d[s])
            res[s]['test'].to_excel(writer, d[s], startcol=3)
            eva = eva.append(res[s]['eva'])
        eva = eva[['type', 'EVS', 'MAE', 'MSE', 'R2']]
        eva.to_excel(writer, d['eva'])
        
    """    
    model_dir = os.path.join(os.path.abspath('.'), '%s_pkl'%state)
    model_name = 'GradientBoostingRegressor.pkl'
    model_name = os.path.join(model_dir, model_name)
    model = ut.load_model(model_name)
    print(model)
    data_x = data_x[0:1]
    data_y = data_y[0:1]
    print(data_x, data_y)
    res0 = ut.valid_model(model, data_x, data_y, feature_method='f_regression')
    print(res)
    """
if __name__ == '__main__':
    main()
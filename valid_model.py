#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:17:18 2018

@author: wuzhiqiang
"""
import os
import build_model as bm
import utils as ut
import preprocess_data as ppd
import read_data as rd
import sql_operation as sql
import pymysql
import time
import pandas as pd
import random

def get_time(data, index):
    start_time = data['start_time'].iloc[index]
    end_time = data['end_time'].iloc[index]
    return start_time, end_time

def get_score_data(file_name, cell_no):
    score_data = pd.read_csv(file_name+'.csv', encoding='gb18030')
    score_data = score_data[score_data['cell_no'] == int(cell_no)]
    return score_data
    
def get_score(score_data, mode='fixed'):
    if mode == 'fixed':
        index = 2
    elif mode == 'random':
        index = random.randint(0, len(score_data))
    score = score_data['c'].iloc[index]
    start_time, end_time = get_time(score_data, index)
    return score, start_time, end_time
        
def get_cell_data(config, cell_no, start_time, end_time, **kwg):
    """
    #kwg包含数据切片要求
    #interval 数据点数量
    """
    table_name = 'cell_'+cell_no
    conn = sql.create_connection(config)
    cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
    start = time.time()
    sql_cmd = "select * from %s where stime between '%s' and '%s'"\
                %(table_name, start_time, end_time)
    cursor.execute(sql_cmd)#获取数据行数
    rows = cursor.fetchall()
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        end = time.time()
        interval = {}
        index = {}
        data_dict = {}
        print('Finished read the data from the database which took %d seconds.'%(end-start))
        if 'interval' in kwg:
            interval = kwg['interval']
        else:
            interval['default'] = len(df)
        if 'index' in kwg:
            index = kwg['index']
        else:
            index['default'] = 0
        for key, value in index.items():
            for k, v in interval.items():
                v = min(len(df), v)
                data_dict[key+'_'+k] = df[value:(value+v)].copy(deep=True)
        return data_dict
    else:
         return None

def get_slip_index(soc, total_num):
    """
    #根据给定的soc区间获得数据切片位置
    """
    soc = min(100, soc)
    index = total_num * 100 // soc
    return index

def get_x_y_data(data_dict):
    """
    """
    
def main():
    state = 'discharge'
    file_dir = os.path.join(os.path.abspath('.'), 'data')
    file_name = r'processed_%s_data'%state
    cell_no = '15'
    config = {'s': 'localhost', 'u': 'root', 'p': 'wzqsql', 'db': 'cell_lg36',
                  'port': 3306}
    index_list = {'100soc':0, '90soc':300, '80soc':600, '70soc':900, '60soc':1200}
    interval_list = {'5mins':300, '10mins':600}
    score_data = get_score_data(os.path.join(file_dir, file_name), cell_no)
    score, start_time, end_time = get_score(score_data)
    data_dict = get_cell_data(config, cell_no, start_time, end_time,
                              interval=interval_list, index=index_list)
    df = pd.DataFrame()
    for key, value in data_dict.items():
        #
        data = ppd.preprocess_data(file_dir, 'valid_'+file_name, cell_no, data0=value)
        data['c'] = score
        df = df.append(data)
    rd.save_data_csv(df, 'valid_'+file_name, file_dir)
    file_name = os.path.join(file_dir, r'valid_processed_%s_data.csv'%state)
    data_x, data_y = bm.calc_feature_data(file_name, data=df)
    model_dir = os.path.join(os.path.abspath('.'), '%s_pkl'%state)
    model_name = 'GradientBoostingRegressor.pkl'
    model_name = os.path.join(model_dir, model_name)
    model = ut.load_model(model_name)
    res = ut.valid_model(model, data_x, data_y, feature_method='f_regression')
    res.index = data_dict.keys()
    print(res)
    output = os.path.join(os.path.join(os.path.abspath('.'), 'result'), 'vaild_%s_result.csv'%state)
    res.to_csv(output, mode='a', encoding='gb18030')

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:58:07 2018

@author: wuzhiqiang
"""

import os
import preprocess_data as ppd
import read_data as rd
import sql_operation as sql
import pymysql
import time
import pandas as pd

def get_time(ser):
    start_time = ser['start_time']
    end_time = ser['end_time']
    return start_time, end_time

def get_raw_data(file_name):
    raw_data = pd.read_csv(file_name+'.csv', encoding='gb18030')
    return raw_data
    
def get_info(ser):
    score = ser['c']
    start_time, end_time = get_time(ser)
    print(score, start_time, end_time)
    return score, start_time, end_time
        
def get_cell_data(config, cell_no, start_time, end_time):
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
    end = time.time()
    print('Finished read the data from the database which took %d seconds.'%(end-start))
    if len(rows) > 0:
        df = pd.DataFrame(rows)
        scale = len(df)
        interval = scale // 10
        data_dict = {}
        for i in range(0, scale, interval):
            for j in range(interval, scale, interval):
                if (i+j) <= scale:
                    data_dict[str(i)+'_'+str(j)] = df[i:(i+j)].copy(deep=True)
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

def scale_data(ser, config):
    score, start_time, end_time = get_info(ser)
    cell_no = str(ser['cell_no'])
    data_dict = get_cell_data(config, cell_no, start_time, end_time)
    df = pd.DataFrame()
    for key, value in data_dict.items():
        if len(value) > 10:
            data = ppd.preprocess_data('unused', 'unused', cell_no, data0=value)
            data['c'] = score
            df = df.append(data)
    return df
    
def generate_data(raw_data, config):
    """
    """
    start = time.time()
    for i in range(len(raw_data)):
        #选择要扩展的种子
        print('-------------%d--------------'%i)
        new_data = scale_data(raw_data.iloc[i], config)
        raw_data = raw_data.append(new_data)
    raw_data = raw_data.reset_index(drop=True)
    end = time.time()
    print('Finished generate date which took %d seconds.'%(end-start))
    return raw_data
    
def main():
    state = 'discharge'
    file_dir = os.path.join(os.path.abspath('.'), 'data')
    file_name = r'processed_%s_data'%state
    config = {'s': 'localhost', 'u': 'root', 'p': 'wzqsql', 'db': 'cell_lg36',
                  'port': 3306}
    raw_data = get_raw_data(os.path.join(file_dir, file_name))
    data = generate_data(raw_data, config)
    rd.save_data_csv(data, 'scale_'+file_name, file_dir)
        

if __name__ == '__main__':
    main()
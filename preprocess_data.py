#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 10:00:21 2018

@author: wuzhiqiang
"""
print(__doc__)
import read_data as ppd
import os
import pandas as pd
import numpy as np
import time
from dateutil import parser

def calc_other_vectors(df):
    """
    #计算dq/dv值
    #由于没有dq值，因此使用i代替
    #计算i/dv
    """
    df['dqdv'] = df['current'] / df['voltage'].diff()
    if (df['current'].mean(skipna=True)) > 0: #充电
        df['c'] = df['charge_c']
        df['e'] = df['charge_e']
    elif (df['current'].mean(skipna=True)) < 0: #放电
        df['c'] = df['discharge_c']
        df['e'] = df['discharge_e']
    else:
        df['c'] = 0
        df['e'] = 0
    return df

def slip_data(df):
    """
    """
    PROCESS_GAP = 10 #10points,10sec
    PROCESSING_GAP = 10
    data0 = pd.DataFrame()
    for value in set(df['step_no'].tolist()):#到所有都处理完了最后再做一次排序
        idx = df[df['step_no'] == value].index
        data = pd.DataFrame()
        j_last = 0
        cnt = 0
        for j in range(1, len(idx) + 1):
            if j >= len(idx) or idx[j] - idx[j - 1] > PROCESS_GAP:    
                cur_df = df.loc[idx[j_last]:idx[j-1]] #idx[x]代表df的index值,所以用loc
                print('clip %d : j: %d -> %d, the length of cur_df: %d.'
                      %(cnt, idx[j_last], idx[j-1], len(cur_df)))
                j_last = j
                if len(cur_df) < PROCESSING_GAP:
                    continue
                cur_df = calc_other_vectors(cur_df)
                data = data.append(transfer_data(cnt, cur_df))
                cnt += 1
        data0 = data0.append(data)
    return data0
                
def preprocess_data(data_dir, filename, data0=None):
    """
    将预处理后的数据按一次充电过程进行分割合并
    """
    DROPNA_THRESH = 5
    if data0 is None:
        file = os.path.join(data_dir, filename)
        start = time.time()
        data0 = pd.read_csv(file+'.csv', encoding='gb18030')
        end = time.time()
        print('Done, it took %d seconds to read the data.'%(end-start))
    
    #filt samples on rows, if a row has too few none-nan value, drop it
    data0 = data0.dropna(thresh=DROPNA_THRESH)
    data = pd.DataFrame()

    #regular the cycle_no
    temp = data0[['cycle_no']]
    j_last = 0
    cnt = 0  
    for j in range(1, len(temp) + 1):
        if j >= len(temp) or temp['cycle_no'].iloc[j] < temp['cycle_no'].iloc[j - 1]:#下一个实验的循环计数开始
            if j_last == 0:
                bias = 0
            else:
                bias = data0['cycle_no'].iloc[j_last - 1]
            cur_df = data0.iloc[j_last:j]  
            cur_df['cycle_no'] = cur_df['cycle_no'] + bias      
            print('clip %d : j: %d -> %d, the length of cur_df: %d.'
                      %(cnt, j_last, j, len(cur_df)))
            j_last = j
            data = data.append(slip_data(cur_df))
    data['start_time'] = data['start_time'].apply(lambda x: parser.parse(x))
    data = data.sort_values('start_time')
    #data['start_time'] = data['start_time'].apply(str)
    data = data.reset_index(drop=True)
    return data

def transfer_data(cnt, cur_df):
    """
    将2维的df转换为1维
    """
    df = pd.DataFrame(columns=['start_time', 'end_time',
                               'data_num'])
    df.loc[cnt, 'start_time'] = cur_df['time'].iloc[0]
    df.loc[cnt, 'end_time'] = cur_df['time'].iloc[-1]
    df.loc[cnt, 'data_num'] = len(cur_df)
    df.loc[cnt, 'c'] = cur_df['c'].iloc[-1]
    df.loc[cnt, 'e'] = cur_df['e'].iloc[-1]
    
    for col_name in cur_df.columns:
        for fix in ['voltage', 'current', 'dqdv']:
            if fix in col_name:
                cal_stat_row(cnt, cur_df[col_name], col_name, df)
                cal_stat_row(cnt, cur_df[col_name].diff(), col_name + '_diff', df)
                cal_stat_row(cnt, cur_df[col_name].diff().diff(), col_name + '_diff2', df)
                cal_stat_row(cnt, cur_df[col_name].diff() / cur_df[col_name], col_name + '_diffrate', df)
        if 'temperature' in col_name:
            cal_stat_row(cnt, cur_df[col_name], col_name, df)
    return df

def cal_stat_row(cnt, ser, col_name, df):
    """
    求统计值
    """
    func = lambda x: x.fillna(method='ffill').fillna(method='bfill').dropna()
    ser = ser.replace(np.inf, np.nan)
    ser = ser.replace(-np.inf, np.nan)
    ser = func(ser)
    df.loc[cnt, col_name + '_mean'] = ser.mean(skipna=True)
    df.loc[cnt, col_name + '_min'] = ser.min(skipna=True)
    df.loc[cnt, col_name + '_max'] = ser.max(skipna=True)
    df.loc[cnt, col_name + '_median'] = ser.median(skipna=True)
    df.loc[cnt, col_name + '_std'] = ser.std(skipna=True)
    
def main():
    data_ori_dir = os.path.normpath('/Users/admin/Documents/data/电池数据')
    data_dir = os.path.join(os.path.abspath('.'), 'data')
    cell_no = '12'
    temperature = '25'
    cycle = '0-1000'
    filename = 'LG36-%s-%s_%s'%(temperature, cell_no, cycle)
    #"""
    data_ori = ppd.read_data(data_ori_dir, cell_no, temperature)
    data = ppd.clean_data(data_ori)
    ppd.save_data_csv(data, filename, data_dir, 500000)
    #"""
    data = preprocess_data(data_dir, filename)
    ppd.save_data_csv(data, 'processed_'+filename, data_dir)
    
if __name__ == '__main__':
    main()
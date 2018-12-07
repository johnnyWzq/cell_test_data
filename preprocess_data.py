#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 10:00:21 2018

@author: wuzhiqiang
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
from read_data import save_data_csv

def preprocess_data(data_dir, filename):
    """
    将预处理后的数据按一次充电过程进行分割合并
    """
    CHARGE_TIMEGAP = 300  # 300 seconds = 5 minutes
    CHARGING_TIMEGAP = 60
    CA_KWH_UB = 350.0
    DROPNA_THRESH = 5
    
    file = os.path.join(p_data_dir, filename)
    start = time.time()
    data = pd.read_csv(file+'.csv', encoding='gb18030')
    end = time.time()
    print('Done, it took %d seconds to read the data.'%(end-start))
    
    #filt samples on rows, if a row has too few none-nan value, drop it
    data = data.dropna(thresh=DROPNA_THRESH)
    
     # group by bms_id and sort by time
    data_gp = data.groupby('bms_id')
    #data = pd.DataFrame(columns=['bms_id', 'start_time', 'end_time',
    #                             'charger_id', 'data_num'])
    data = pd.DataFrame()
    cnt = 0
    num = 0
    print('The numbers of the data clips is: %d'%len(processed_data_gp.groups))
    for i in processed_data_gp.groups:
        df = processed_data_gp.get_group(i) #第i组
        print('NO.%d: '%num, i, df.shape, cnt)
        num += 1
        df = df.sort_values('time')
        j_last = 0
        for j in range(1, len(df) + 1):
            if j >= len(df) or (df.iloc[j]['time'] - df.iloc[j - 1]['time']).seconds > CHARGE_TIMEGAP:
                    
                if j >= len(df):
                    cur_df = df.iloc[j_last:]
                elif (df.iloc[j]['time'] - df.iloc[j - 1]['time']).seconds > CHARGE_TIMEGAP:
                    cur_df = df.iloc[j_last:j]
                    #j_last = j

                func = lambda x: x.fillna(method='ffill').fillna(method='bfill').dropna()
                cur_df = func(cur_df)
                
                print('clip %d : j: %d -> %d, the length of cur_df: %d.'
                      %(cnt, j_last, j, len(cur_df)))
                #print('j:', j_last, '->', j, 'len(cur_df):', tmp_len, '->', len(cur_df), 'cnt=', cnt)
                j_last = j
                if len(cur_df) <= 0 or (cur_df['time'].iloc[-1] - cur_df['time'].iloc[0]).seconds < CHARGING_TIMEGAP:
                    continue
                cur_df['score_ca_kwh'] = cur_df['cp_kwh'].diff() / cur_df['bp_soc'].diff() * 100
                cur_df['score_ca_ah'] = cur_df['cp_ah'].diff() / cur_df['bp_soc'].diff() * 100
                cur_df['score_ca_health'] = cur_df['health']
                
                data = data.append(transfer_data(cnt, i, cur_df))
                cnt += 1

    return data

def transfer_data(cnt, bms_id, cur_df):
    """
    将2维的df转换为1维
    """
    df = pd.DataFrame(columns=['bms_id', 'start_time', 'end_time',
                                 'charger_id', 'data_num'])
    df.loc[cnt, 'bms_id'] = bms_id
    df.loc[cnt, 'start_time'] = cur_df['time'].iloc[0]
    df.loc[cnt, 'end_time'] = cur_df['time'].iloc[-1]
    df.loc[cnt, 'charger_id'] = cur_df['charger_id'].iloc[0]
    df.loc[cnt, 'data_num'] = len(cur_df)
    
    for col_name in cur_df.columns:
        for fix in ['bi_']:
            if fix in col_name:
                df.loc[cnt, col_name + '_mean'] = cur_df[cur_df[col_name] > 0][col_name].mean(skipna=True)#选择col_name大于0的行并求平均值
        for fix in ['score_']:
            if fix in col_name:
                cal_stat(cnt, cur_df[col_name], col_name, df)
        for fix in ['cp_', 'bp_', '_sv', '_st']:
            if fix in col_name:
                cal_stat(cnt, cur_df[col_name], col_name, df)
                cal_stat(cnt, cur_df[col_name].diff().abs(), col_name + '_diff', df)
                cal_stat(cnt, cur_df[col_name].diff().diff().abs(), col_name + '_diff2', df)
                cal_stat(cnt, cur_df[col_name].diff().abs() / cur_df[col_name], col_name + '_diffrate', df)
    return df

def cal_stat(cnt, ser, col_name, df):
    """
    求统计值
    """
    df.loc[cnt, col_name + '_mean'] = ser.mean(skipna=True)
    df.loc[cnt, col_name + '_min'] = ser.min(skipna=True)
    df.loc[cnt, col_name + '_max'] = ser.max(skipna=True)
    df.loc[cnt, col_name + '_median'] = ser.median(skipna=True)
    df.loc[cnt, col_name + '_std'] = ser.std(skipna=True)
    
def main():
    data_dir = os.path.join(os.path.abspath('.'), 'data')
    cell_no = '14'
    temperature = '25'
    cycle = '0-1000'
    filename = 'LG36-%s-%s_%s'%(temperature, cell_no, cycle)
    
    data = preprocess_data(data_dir, filename)
    save_data_csv(data, 'processed'+filename, data_dir)
    
if __name__ == '__main__':
    main()
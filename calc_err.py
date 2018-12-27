#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 10:31:00 2018

@author: wuzhiqiang
"""
import os
import pandas as pd

state = 'discharge'
file_dir = os.path.join(os.path.abspath('.'), 'data')
file_name = os.path.join(os.path.join(file_dir, r'scale_processed_%s_data.csv'%state))
result_dir = os.path.join(os.path.abspath('.'), 'g_result')
df0 = pd.read_csv(file_name, encoding='gb18030')
df1 = pd.read_excel(os.path.join(result_dir, '%s_result.xlsx'%state), sheet_name='随机森林', usecols=[3,4,5])
df1 = df1.dropna()
df0 = df0.loc[df1.index][['data_num','cell_no']]
df=pd.merge(df0, df1, left_on=df0.index, right_on=df1.index)
df['delta_time'] = df['data_num'] // 60
df['err%'] = ((df['test:pred_y'] - df['test:true_y']) / df['test:true_y']) * 100
df['abs_err%'] = df['err%'].abs()
df.to_excel(os.path.join(result_dir, '%s_result_analys.xlsx'%state))
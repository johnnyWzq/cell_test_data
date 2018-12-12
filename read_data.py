#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 01:55:04 2018

@author: wuzhiqiang
"""

print(__doc__)
import os
import re
import time
from dateutil import parser
import pandas as pd

def read_excel(filename):
    print('reading a excel file...')
    temp = pd.DataFrame()
    start = time.time()
    data_dict = pd.read_excel(filename, sheet_name=None, 
                              nrows=66000, encoding='gb18030')
    for key, value in data_dict.items():
        if re.match(r'Channel_\d+-\d+_\d+', key):
            temp = temp.append(value, ignore_index=True)
    end = time.time()
    print('Done, it took %d seconds to read the data.'%(end-start))
    print('data shape: ' + temp.shape.__str__())
    return temp

def read_excel_files(data_dir, regx):
    temp = []
    for filename in os.listdir(data_dir):#获取文件夹内所有文件名
        if re.match(regx, filename):
            print('---------------------------------------')
            print(filename)
            temp.append(read_excel(os.path.join(data_dir, filename)))
    temp = pd.concat(tuple(temp), ignore_index=True)
    print('data shape: ' + temp.shape.__str__())
    return temp

def read_data(data_dir, cell_no='14', temperature='25'):
    """
    读取数据
    """
    if os.path.exists(data_dir):
        regx = r'LG36-%sdu-[0-9a-zA-Z\_]+-[0-9a-zA-Z]+-[0-9a-zA-Z]+-[0-9a-zA-Z]+-%s.\w+'\
                %(temperature, cell_no)
        data = read_excel_files(data_dir, regx)
        if 'Temperature (C)_1' in data.columns:
            data = data.rename(columns = {'Temperature (C)_1': 'temperature'})
        else:
            data['temperature'] = int(temperature)
        return data
    else:
        print("There isn't such a path.")
        return None

def clean_data(data):
    """
    对读取对数据进行初步对清洗
    """
    
    #对每一张表进行清洗
    #删除无用的列
    if data is None:
        print('The data is None.')
        return None
    print('cleaning cell data...')
    data = data.rename(columns = {'Test_Time(s)': 'timestamp', 'Date_Time': 'time',
                                  'Step_Index': 'step_no', 'Cycle_Index': 'cycle_no',
                                  'Current(A)': 'current', 'Voltage(V)': 'voltage',
                                  'Charge_Capacity(Ah)': 'charge_c',
                                  'Discharge_Capacity(Ah)': 'discharge_c',
                                  'Charge_Energy(Wh)': 'charge_e', 'Discharge_Energy(Wh)': 'discharge_e',
                                  'dV/dt(V/s)': 'dv/dt'})
    data = data[['timestamp', 'time', 'cycle_no', 'step_no', 'current', 'voltage',
                 'charge_c', 'discharge_c',
                 'charge_e', 'discharge_e', 'dv/dt', 'temperature']]
    data = data.dropna()
    data['time'] = data['time'].apply(str)
    data['time'] = data['time'].apply(lambda x: parser.parse(x))
    data = data.sort_values('time')
    data['time'] = data['time'].apply(str)
    return data
    
def save_data_xlsx(data, max_lens, filename, output_dir=None):
    """
    保存数据，如果数据行较大，进行分割分别放入sheet
    """
    if data is None:
        print('The data is None.')
    elif output_dir:
        writer = pd.ExcelWriter(os.path.join(output_dir, '%s.xlsx'%filename))
        lens = data.shape[0]
        i = 0
        j = lens
        while j > 0:
            #切片，并保存至sheet
            k = i + min(j, max_lens)
            df = data[i:k]
            #保存处理后的数据
            df.to_excel(writer, '%d-%d'%(i,k))
            print('Saved the slip data in a sheet which named %d-%d.'%(i,k))
            i = i + max_lens
            j = j - max_lens
        print('The whole data has been saved.')
        
def save_data_csv(data, filename, output_dir=None, max_lens=None):
    """
    保存数据
    """
    if data is None:
        print('The data is None.')
    elif output_dir:
        start = time.time()
        data.to_csv(os.path.join(output_dir, '%s.csv'%filename), chunksize=max_lens,
                    index=False, index_label=False, encoding='gb18030')
        #data[:70000].to_csv(os.path.join(output_dir, 'test_%s.csv'%filename), index=False, index_label=False, encoding='gb18030')
        print('The whole data has been saved.')
        end = time.time()
        print('Done, it took %d seconds to save the data.'%(end-start))
    
def save_workstate_data(regx, data_dir):
    temp = []
    start = time.time()
    for filename in os.listdir(data_dir):#获取文件夹内所有文件名
        if re.match(regx, filename):
            print('---------------------------------------')
            print(filename)
            temp.append(pd.read_csv(os.path.join(data_dir, filename),
                                    encoding='gb18030'))
    temp = pd.concat(tuple(temp), ignore_index=True)
    print('data shape: ' + temp.shape.__str__())
    end = time.time()
    print('Done, it took %d seconds to read the data.'%(end-start))
    data = temp[temp['current_mean'] == 0].reset_index(drop=True) #静置数据
    save_data_csv(data, 'processed_rest_data', data_dir)
    data = temp[temp['current_mean'] > 0].reset_index(drop=True) #充电数据
    save_data_csv(data, 'processed_charge_data', data_dir)
    data = temp[temp['current_mean'] < 0].reset_index(drop=True) #放电数据
    save_data_csv(data, 'processed_discharge_data', data_dir)
    
def test(p_data_dir, filename):
    file = os.path.join(p_data_dir, filename)
    start = time.time()
    data = pd.read_csv(file+'.csv', encoding='gb18030')
    end = time.time()
    print('Done, it took %d seconds to read the data.'%(end-start))
    print(data.shape)
    """
    start = time.time()
    data = pd.read_excel(file+'.xlsx', encoding='gb18030')
    end = time.time()
    print('Done, it took %d seconds to read the data.'%(end-start))
    """
           
def main():
    data_dir = os.path.normpath('/Users/admin/Documents/data/电池数据')
    p_data_dir = os.path.join(os.path.abspath('.'), 'data')
    cell_no = '14'
    temperature = '25'
    cycle = '0-1000'
    filename = 'LG36-%s-%s_%s'%(temperature, cell_no, cycle)
    #test(p_data_dir, filename)
    
    data_ori = read_data(data_dir, cell_no, temperature)
    data = clean_data(data_ori)
    save_data_csv(data, filename, p_data_dir, 500000)
    
if __name__ == '__main__':
    main()
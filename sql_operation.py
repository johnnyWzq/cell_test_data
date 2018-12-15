#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:14:27 2018

@author: wuzhiqiang
"""
import re
import pymysql
from sqlalchemy import create_engine
from sqlalchemy.types import NVARCHAR, Float, Integer

def create_connection(config):
    
    print('conneting the database...')
    try:
        
        conn = pymysql.connect(host=config['s'], user=config['u'],
                               password=config['p'], db=config['db'], 
                               port=3306, charset='utf8')
    
    except pymysql.OperationalError:
        print ("error: Could not Connection SQL Server!please check your dblink configure!")
        return None
    else:
        print('the connetion is sucessful.')
        return conn
    
def close_connection(conn):
    """
    close the connection
    """
    conn.close()
    print('the connetion is closed.')
    
def table_exists(conn,table_name):
    """
    #用来判断表是否存在
    """
    cursor = conn.cursor()
    sql_cmd = "show tables;"
    cursor.execute(sql_cmd)
    tables = [cursor.fetchall()]
    table_list = re.findall('(\'.*?\')',str(tables))
    table_list = [re.sub("'",'',each) for each in table_list]
    if table_name in table_list:
        return 1        #存在返回1
    else:
        return 0        #不存在返回0

def create_table(conn, sql_cmd, table_name):
    """
    """
    if(table_exists(conn, table_name) != 1):
        print("it can be create a table named bat_info.")
        cursor = conn.cursor()
        cursor.execute(sql_cmd)
        cursor.close()
        
def create_sql_engine(basename, user, password, server, port):
    return create_engine('mysql+pymysql://%s:%s@%s:%s/%s?charset=utf8'
                  %(user, password, server, port, basename))
    
def mapping_df_types(df):
    dtypedict = {}
    for i, j in zip(df.columns, df.dtypes):
        if "object" in str(j):
            dtypedict.update({i: NVARCHAR(length=255)})
        if "float" in str(j):
            dtypedict.update({i: Float(precision=8, asdecimal=True)})
        if "int" in str(j):
            dtypedict.update({i: Integer()})
    return dtypedict
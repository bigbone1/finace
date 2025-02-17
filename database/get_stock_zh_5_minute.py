from time import sleep
import akshare as ak
import pandas as pd
import sys
sys.path.append(r'D:\python\finace')
from libs import MyEngine, RepeatRun
import baostock as bs
import pandas as pd

def save_stock_zh_a_minute(symbol):
    def _core(symbol):
        print(symbol)
        rs = bs.query_history_k_data_plus(symbol,
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date='2023-01-01', end_date='2024-09-14',
            frequency="5", adjustflag="1")
        print('query_history_k_data_plus respond error_code:'+rs.error_code)
        print('query_history_k_data_plus respond  error_msg:'+rs.error_msg)

        #### 打印结果集 ####
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            data_list.append(rs.get_row_data())
        result = pd.DataFrame(data_list, columns=rs.fields)
        return result

    int_symbol = symbol
    str_symbol = str(int_symbol).rjust(6, '0')
    # print(str_symbol)
    result = _core(f"sh.{str_symbol}")
    if result.shape[0] == 0:
        result = _core(f"sz.{str_symbol}")
    print(result.shape)
    MyEngine().to_mysql(result, 'stock_zh_5_minute', index=False)

if __name__ == '__main__':
    ### 登陆系统 ####
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    sql = 'select 代码 from industry_cons_em_details'
    symbols = MyEngine().read_sql_query(sql)['代码'].unique()
    print(symbols.shape[0], 'all number')

    sql1 = 'select distinct code from stock_zh_5_minute'
    symbols_exist = MyEngine().read_sql_query(sql1)['code'].apply(lambda x: int(x.split('.')[1])).unique()

    target_symbols = set(symbols) - set(symbols_exist)
    print(len(target_symbols), 'target number')
    for i in target_symbols:
        # sleep(1)
        save_stock_zh_a_minute(i)
    bs.logout()
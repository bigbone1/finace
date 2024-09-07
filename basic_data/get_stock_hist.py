import akshare as ak
import pandas as pd
import sys
sys.path.append(r'D:\python\finace')
from libs import MyEngine, RepeatRun, convert_names



def save_stock_zh_a_hist(symbol):
    adjust = 'hfq'
    period = 'daily'
    symbol = str(symbol).rjust(6, '0')
    df = ak.stock_zh_a_hist(symbol=symbol, adjust=adjust, period=period)
    df = convert_names(df)
    df['adjust'] = adjust
    df['period'] = period
    df['stock_code'] = df['stock_code'].astype(int)
    MyEngine().to_mysql(df, 'stock_zh_a_hist_daily', index=False)


if __name__ == '__main__':
    sql = 'select 代码 from industry_cons_em_details'
    symbols = MyEngine().read_sql_query(sql)
    sql2 = 'select distinct stock_code FROM stock_zh_a_hist_daily'
    exists_symbols = MyEngine().read_sql_query(sql2)
    symbols = symbols[~symbols['代码'].isin(exists_symbols['stock_code'].astype(int))]['代码'].unique()

    print(symbols.shape[0], 'all number')
    print(exists_symbols.shape[0], 'exist number')
    print(symbols.shape[0], 'waiting number')

    error_list = RepeatRun(max_workers=10).run(save_stock_zh_a_hist, symbols)
    pd.DataFrame({'error_sysbol': error_list}).to_csv('stock_zh_a_hist_daily.csv', index=False)

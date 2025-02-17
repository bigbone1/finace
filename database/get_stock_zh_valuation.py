import akshare as ak
import pandas as pd
import sys
sys.path.append(r'D:\python\finace')
from libs import MyEngine, RepeatRun



def save_stock_zh_valuation_baidu(symbol):
    int_symbol = symbol
    str_symbol = str(symbol).rjust(6, '0')
    res_list = []
    names_en = ['market_capitalization', 'earnings_ratio_ttm', 'earnings_ratio_static', 'book_ratio', 'cash_flow_ratio']
    for i, name_cn in enumerate(["总市值", "市盈率(TTM)", "市盈率(静)", "市净率", "市现率"]):
        df = ak.stock_zh_valuation_baidu(symbol=str_symbol, indicator=name_cn, period='全部')
        df.rename(columns={'value': names_en[i]}, inplace=True)
        res_list.append(df)
    if len(res_list) == 5:
        res_df = res_list[0]
        for i in res_list[1:]:
            res_df = pd.merge(res_df, i, on='date')
        res_df['stock_code'] = int_symbol
        MyEngine().to_mysql(res_df, 'stock_zh_valuation_baidu', index=False)
    else:
        raise ValueError('计算次数错误')

if __name__ == '__main__':
    sql = 'select 代码 from industry_cons_em_details'
    symbols = MyEngine().read_sql_query(sql)['代码'].values
    # sql2 = 'select distinct stock_code FROM stock_zh_a_hist_daily'
    # exists_symbols = MyEngine().read_sql_query(sql2)
    # symbols = symbols[~symbols['代码'].isin(exists_symbols['stock_code'].astype(int))]['代码'].unique()

    print(symbols.shape[0], 'all number')
    # print(exists_symbols.shape[0], 'exist number')
    # print(symbols.shape[0], 'waiting number')

    error_list = RepeatRun(max_workers=10, sleep_seconds=5).run(save_stock_zh_valuation_baidu, symbols)
    pd.DataFrame({'error_sysbol': error_list}).to_csv('stock_zh_valuation_baidu.csv', index=False)

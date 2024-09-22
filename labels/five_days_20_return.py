"""
输入：days = 5; max_return_days = 5; max_return = 20
5日后存在20%收益的股票
输出格式：code, start_date, end_date, label 
"""
import code
from libs.mysql import MyEngine
import numpy as np
import pandas as pd
from numpy_ext import rolling_apply

class HighFreqLabel():
    def __init__(self, start_date, end_date, target_code_num,
                 input_days=5, max_return_days=5, max_return=20):
        self.start_date = start_date
        self.end_date = end_date
        self.target_code_num = target_code_num    # -1: 所有股票
        self.input_days = input_days   
        self.max_return_days = max_return_days   # 计算回报的天数 尾盘最后半小时买入
        self.max_return_thresh = max_return      # 回报最小阈值20%

    @property
    def target_codes(self):
        return self._target_codes
    
    @property
    def engine(self):
        self._engine = MyEngine()
        return self._engine

    def get_labels(self, multi=False):
        self._get_target_codes()

        sql = self.engine.create_sql(
            columns=['stock_code', 'date', 'open_price', 'close_price', 'high_price', 'low_price'],
            table_name='stock_zh_a_hist_daily',
            start_date=self.start_date,
            end_date=self.end_date,
            codes=self._target_codes
        )
        data = self.engine.read_sql_query(sql)

        if multi:
            data = self._handle_data_multi(data)
        else:
            data = self._handle_data_sin(data)

        data['input_days'] = self.input_days
        data['max_return_thresh'] = self.max_return_thresh
        data['max_return_days'] = self.max_return_days
        data.drop(columns=['open_price', 'close_price', 'high_price', 'low_price'], inplace=True)
        self.engine.to_mysql(data, f'high_freq_labels')
    
    def _get_target_codes(self):
        # 获取目标股票代码
        engine = MyEngine()
        sql = engine.create_sql(
            ['stock_code'], 
            'stock_zh_a_hist_daily', self.start_date, self.end_date)
        codes = engine.read_sql_query(sql)
        codes['stock_code'] = codes['stock_code'].astype(str)
        self._target_codes = codes['stock_code'].unique()[:self.target_code_num]
        self.target_code_num = len(self._target_codes)
    
    def _handle_data_sin(self, data: pd.DataFrame):
        def apply_func(h, c):
            try:
                res = (np.max(h)-c.iloc[0])/c.iloc[0]
            except:
                res = np.nan 
            return res 
        
        res = []
        for code, df in data.groupby('stock_code'):
            df.drop_duplicates(inplace=True)
            if df.shape[0] < self.input_days*10:
                continue
            df.sort_values(by=['date'], inplace=True)
            df.reset_index(inplace=True, drop=True)
            # res['date'] = pd.to_datetime(res['date'])
            df['end_date'] = df['date']
            df['start_date'] = df['date'].shift((self.input_days-1))
            df['return_end_date'] = df['date'].shift(-1*self.max_return_days)
            df['max_return'] = pd.Series(rolling_apply(apply_func, self.max_return_days+1, df['high_price'], df['close_price'],
                                             n_jobs=10))*100

            df['label'] = df['max_return'].apply(lambda x: 1 if x >= self.max_return_thresh else 0)
            if isinstance(res, list):
                res = df
            else:
                res = pd.concat([res, df])
        # res.dropna(inplace=True)
        return res 
    
    def _handle_data_multi(self, data):
        return data  
    
    # def prepare_data(start_date, end_date, target_code_num, train=True):
        sql3 = f'select stock_code, date, open_price, close_price \
            from stock_zh_a_hist_daily where date between "{start_date}" and "{end_date}"'
        predicts = MyEngine().read_sql_query(sql3)
        predicts.sort_values(by=['date'], inplace=True)
        target_code = predicts['stock_code'].unique()[:target_code_num]
        def func_(x):
            return x.iloc[-1]
        predicts = predicts[predicts['stock_code'].isin(target_code)]
        predicts['close_price_next_5'] = predicts.groupby(['stock_code'])['close_price'].transform(lambda x: x.rolling(5).apply(func_))
        predicts['cum_sum_5'] = (predicts['close_price_next_5'] - predicts['open_price']) / predicts['open_price'] * 100

        sql4 = f'select stock_code, date, open_price, close_price, volume, \
            turnover, amplitude, price_change_percentage, price_change_amount, \
                turnover_rate  from stock_zh_a_hist_daily where date between "{start_date}" and "{end_date}" and stock_code in '
        for i, code in enumerate(target_code):
            if i == (len(target_code)-1):
                sql4 = sql4 + str(code) + ')'
            elif i == 0:
                sql4 = sql4 + '(' + str(code) + ', '
            else:
                sql4 = sql4 + str(code) + ', '
        inputs1 = MyEngine().read_sql_query(sql4)

        sql5 = f'select stock_code, date, market_capitalization, earnings_ratio_ttm, \
            earnings_ratio_static, book_ratio, cash_flow_ratio from stock_zh_valuation_baidu \
                where date between "{start_date}" and "{end_date}" and stock_code in '
        for i, code in enumerate(target_code):
            if i == (len(target_code)-1):
                sql5 = sql5 + str(code) + ')'
            elif i == 0:
                sql5 = sql5 + '(' + str(code) + ', '
            else:
                sql5 = sql5 + str(code) + ', '
        inputs2 = MyEngine().read_sql_query(sql5)

        sql6 = 'select 板块名称 as industry_name, 代码 as stock_code from industry_cons_em_details'
        inputs3 = MyEngine().read_sql_query(sql6)

        inputs1.sort_values(by=['stock_code', 'date'], inplace=True)
        inputs1.loc[:, 'date_seconds'] = pd.to_datetime(inputs1['date']).astype('int64')//1e9
        inputs1['price_change_percentage_next_day'] = inputs1['price_change_percentage'].apply(lambda x: 1 if x>5 else 0)
        # inputs1 = inputs1.groupby('stock_code').apply(lambda x: x['open_price']/x['open_price'].iloc[0])
        inputs1.dropna(inplace=True)
        # data1 = inputs1.drop(columns=['date'])
        # earnings = data1['price_change_percentage_next_day']
        # data3 = data1.drop(columns=['price_change_percentage_next_day'])

        data2 = pd.merge(inputs2, inputs3, on='stock_code', how='left')
        data2.sort_values(by=['stock_code', 'date'], inplace=True)
        data2 = data2.groupby('stock_code').apply(lambda x: x.set_index('date').asfreq('D').reset_index().fillna(method='ffill'))
        industry_map = inputs3['industry_name'].drop_duplicates().reset_index().drop('index', axis=1).reset_index().set_index('industry_name').to_dict()['index']
        data2.loc[:, 'industry_name'] = data2['industry_name'].apply(lambda x: industry_map[x])
        data2.loc[:, 'date_seconds'] = pd.to_datetime(data2['date']).astype('int64')//1e9
        data2.drop(columns=['date'], inplace=True)
        data2.reset_index(inplace=True, drop=True)
        temp = pd.merge(inputs1, data2, on=['stock_code', 'date_seconds'], how='left')
        data2 = temp.fillna(method='ffill').fillna(method='bfill')
        data2 = data2[data2['date_seconds'].isin(inputs1.date_seconds)]

        data2.sort_values(by=['stock_code', 'date_seconds'], inplace=True)
        earnings = data2['price_change_percentage_next_day']
        if train:
            data2.drop(columns=['price_change_percentage_next_day', 'date'], inplace=True)
        print(data2.shape, 'data shape')
        return data2, earnings



if __name__ == '__main__':
    label = HighFreqLabel('2023-01-01', '2024-09-30', target_code_num=-1).get_labels()
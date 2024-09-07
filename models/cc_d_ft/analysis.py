# %%
import sys

from networkx import volume
sys.path.append(r'D:\python\finace')
from libs import MyEngine
import pandas as pd
import seaborn as sns
#coding:utf-8
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

# %% [markdown]
# # 因子选股：
# ## 找出股票
# - 交易量翻倍  k =  2
# - 交易额进入前 turnover_rank = 100

# %%
class Params():
    ave_volume_k = 5 #交易量倍数
    ave_volume_days = 5 #交易量天数
    turnover_rank = 5000 # 交易量排名

class StockChooser():
    def __init__(self,  start_date, end_date) -> None:
        """
        依据交易量变化等因子筛选出股票库
        """
        self.strategy = 'factor_feng'
        self.start_date = start_date 
        self.end_date = end_date
        self.params = self._init_params()
        self.candidates = pd.DataFrame(data={'code':[], 'date': []})

    def start(self, **kwargs) -> None:
        data = self._get_data()
        data = self._process_data(data)
        for date, dfg in data.groupby(['date']):
            high_volume_rate = dfg[dfg['volume_rate']>self.params.ave_volume_k]['stock_code'].values
            dfg.sort_values(by=['turnover'], inplace=True, ascending=False)
            high_turnover = dfg.iloc[:self.params.turnover_rank, :]['stock_code'].values
            res_code = [i for i in high_volume_rate if i in high_turnover]
            temp  = pd.DataFrame(data={'code':res_code, 'date': [date for i in range(len(res_code))] })
            self.candidates = pd.concat([self.candidates, temp], axis=0)

    def _init_params(self) -> Params:
        return Params()

    def __str__(self) -> str:
        return self.strategy

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data.sort_values(by=['stock_code', 'date'], inplace=True, ascending=True)
        data['volume_next_day'] = data.groupby(['stock_code'])['volume'].shift(-1)
        data['volume_ave'] = data.groupby(['stock_code'])['volume'].rolling(self.params.ave_volume_days).mean()
        data['close_price_next_days'] = data.groupby(['stock_code'])['close_price'].shift(-5)
        data.dropna(inplace=True)
        data['cumsum_return_next_days'] = (data['close_price_next_days']-data['close_price'])/data['close_price']*100
        data['volume_rate'] = data['volume_next_day'] / data['volume_ave']
        return data 

    def _get_data(self) -> pd.DataFrame:
        """
        Get the daily stock data from database.

        The data includes stock code, date, volume, open price, close price, turnover, turnover rate and price change percentage.

        Returns
        -------
        pd.DataFrame
            The daily stock data from database.
        """
        # The SQL query to get the daily stock data from database.
        sql = f'''
        select stock_code, date, volume, open_price, close_price, turnover, turnover_rate, price_change_percentage 
        from stock_zh_a_hist_daily 
        where date between "{self.start_date}" and "{self.end_date}"
        '''
        data = MyEngine().read_sql_query(sql)
        return data 


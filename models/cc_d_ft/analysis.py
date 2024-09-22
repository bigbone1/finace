# %%
import os
import sys
from typing import List
from libs import MyEngine
import pandas as pd
import seaborn as sns
#coding:utf-8
import matplotlib.pyplot as plt
import mplfinance as mpf
from pathlib import Path
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

class Params():
    filter_strategy = ['filter_ave_volume', 'filter_up', 'filter_turnover_rate']
    ave_volume_k = 5 #交易量倍数
    ave_volume_days = 5 #交易量天数
    turnover_rank = 5000 # 交易量排名
    turnover_rate = 5
    up = 2  # 上涨幅度
    amplitude = 15
    current_file_path = os.path.abspath(__file__)
    fig_dir = current_dir = os.path.dirname(current_file_path)
    

class StockChooser():
    def __init__(self,  start_date, end_date) -> None:
        """
        依据交易量变化等因子筛选出股票库
        """
        self.strategy = 'factor_feng'
        self.start_date = start_date 
        self.end_date = end_date
        self.params = self._init_params()
        self.candidates = pd.DataFrame(data={'stock_code':[], 'date': []})
    
    @property
    def data_base(self) -> pd.DataFrame:
        return self._base_data
    
    @property
    def data_candidates(self):
        temp= pd.merge(self.data_base, self.candidates, on=['stock_code', 'date'], how='inner')
        return temp.sort_values(by=['cumsum_return_next_days'], ascending=False) 

    def start(self, **kwargs) -> None:
        data = self._get_data()
        data = self._process_data(data)
        self._base_data = data

        for i in self.params.filter_strategy:
            candidate = pd.DataFrame(data={'stock_code':[], 'date': []})
            for date, dfg in data.groupby(['date']):
                res_stock_code = getattr(self, '_'+i)(dfg)
                temp  = pd.DataFrame(data={'stock_code':res_stock_code, 'date': [date[0] for i in range(len(res_stock_code))] })
                candidate = pd.concat([candidate, temp], axis=0)
            
            if self.candidates.empty:
                self.candidates = candidate
            else:
                self.candidates = pd.merge(self.candidates, candidate, on=['stock_code', 'date'], how='inner')

    def _filter_up(self, dfg) -> List:
        up = dfg[dfg['price_change_percentage']>self.params.up]['stock_code'].values
        return up.tolist()
    def _filter_turnover_rate(self, dfg):
        turnover_rate = dfg[dfg['turnover_rate']<self.params.turnover_rate]['stock_code'].values
        return turnover_rate.tolist()
    
    def _filter_amplitude(self, dfg):
        amplitude = dfg[dfg['amplitude']<self.params.amplitude]['stock_code'].values
        return amplitude.tolist()
    
    def _filter_ave_volume(self, dfg) -> List:
        high_volume_rate = dfg[dfg['volume_rate']>self.params.ave_volume_k]['stock_code'].values
        return high_volume_rate.tolist()

    def _filter_turnover_rank(self, dfg) -> List:
        """
        Filter the DataFrame `dfg` based on the turnover rank.

        Parameters:
            dfg (DataFrame): The DataFrame to filter.

        Returns:
            List: A list of stock codes that have high turnover rank.
        """
        dfg.sort_values(by=['turnover'], inplace=True, ascending=False)
        high_turnover = dfg.iloc[:self.params.turnover_rank, :]['stock_code'].values
        return high_turnover.tolist()

    def analysis(self, factors: List[str]):
        # turnover_rate, amplitude
        for f in factors:
            sns.set_theme('notebook')
            ax = sns.scatterplot(data=self.data_candidates, x=f, y='cumsum_return_next_days')
            ax.set_xlabel(f'{f}')
            ax.set_ylabel('return (%)')
            fig_dir = Path(self.params.fig_dir)/'figs'
            fig_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_dir/f'{f}.png')
            plt.clf()
            plt.close()

        # 赚钱概率
        ax = sns.histplot(data=self.data_candidates, x='cumsum_return_next_days')
        ax.set_xlabel('return (%)')
        return_a = self.data_candidates[self.data_candidates['cumsum_return_next_days']>0.5].shape[0] / self.data_candidates.shape[0] * 100
        return_b = self.data_candidates[self.data_candidates['cumsum_return_next_days']>5].shape[0] / self.data_candidates.shape[0] * 100
        return_c = self.data_candidates[self.data_candidates['cumsum_return_next_days']>10].shape[0] / self.data_candidates.shape[0] *100 
        ax.set_title(f'return: {return_a:.1f},{return_b:.1f}, {return_c:.1f}')
        fig_dir = Path(self.params.fig_dir)/'figs'
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_dir/f'hist_volume{self.params.ave_volume_k}_turnover_rate{self.params.turnover_rate}_amplitude{self.params.amplitude}.png')
        plt.clf()
        plt.close()

        # 绘制k线图
        for code in self.data_candidates['stock_code'].unique():
            dfg = self.data_base[self.data_base['stock_code']==code]
            date = self.data_candidates[self.data_candidates['stock_code']==code]['date'].values[0]
            earning = self.data_candidates[self.data_candidates['stock_code']==code]['cumsum_return_next_days'].values[0]
            start_date = date - pd.Timedelta(days=10)
            end_date = date + pd.Timedelta(days=10)
            dfg = dfg[dfg['date'].between(start_date, end_date)]
            daily = dfg[['date', 'open_price', 'close_price', 'high_price', 'low_price', 'volume']]
            daily = daily.rename(columns = {'close_price': 'close',
                                    'open_price': 'open',
                                    'high_price': 'high',
                                    'low_price': 'low'})
            daily.index = daily['date']
            daily = daily.rename(index=pd.Timestamp)
            daily = daily.sort_index()
            fig_dir = Path(self.params.fig_dir)/'figs/K'
            fig_dir.mkdir(parents=True, exist_ok=True)
            mpf.plot(daily, type='candle', volume=True, style='yahoo', returnfig=True,
                               savefig=fig_dir/f'{earning}_{code}.png',
                               mav=(5, 10, 20))
            # plt.clf()
            # plt.close()                                                                                                         ``  

    def _init_params(self) -> Params:
        return Params()

    def __str__(self) -> str:
        return self.strategy

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data.sort_values(by=['stock_code', 'date'], inplace=True, ascending=True)
        data['open_price_next_day'] = data.groupby(['stock_code'])['open_price'].shift(-1)
        data['volume_ave'] = data.groupby('stock_code')['volume'].rolling(window=self.params.ave_volume_days+1, 
                                                                          closed='left').mean().values 
        data['close_price_next_days'] = data.groupby(['stock_code'])['close_price'].shift(-5)
        data['close_price_next_3days'] = data.groupby(['stock_code'])['close_price'].shift(-4)
        data['close_price_next_2days'] = data.groupby(['stock_code'])['close_price'].shift(-3)
        data['close_price_next_5days'] = data.groupby(['stock_code'])['close_price'].shift(-6)
        data['close_price_next_10days'] = data.groupby(['stock_code'])['close_price'].shift(-11)
        data['close_price_next_20days'] = data.groupby(['stock_code'])['close_price'].shift(-21)
        data.dropna(inplace=True)
        data['cumsum_return_next_days'] = (data['close_price_next_days']-data['open_price_next_day'])/data['open_price_next_day'].abs()*100
        data['cumsum_return_next_3days'] = (data['close_price_next_3days']-data['open_price_next_day'])/data['open_price_next_day'].abs()*100
        data['cumsum_return_next_2days'] = (data['close_price_next_2days']-data['open_price_next_day'])/data['open_price_next_day'].abs()*100
        data['cumsum_return_next_5days'] = (data['close_price_next_5days']-data['open_price_next_day'])/data['open_price_next_day'].abs()*100
        data['cumsum_return_next_10days'] = (data['close_price_next_10days']-data['open_price_next_day'])/data['open_price_next_day'].abs()*100
        data['cumsum_return_next_20days'] = (data['close_price_next_20days']-data['open_price_next_day'])/data['open_price_next_day'].abs()*100
        data['volume_rate'] = data['volume'] / data['volume_ave']
        # data['date'] = data.date.astype(str)
        return data 

    def _get_data(self) -> pd.DataFrame:
        """
        Get the daily stock data from database.

        The data includes stock stock_code, date, volume, open price, close price, turnover, turnover rate and price change percentage.

        Returns
        -------
        pd.DataFrame
            The daily stock data from database.
        """
        # The SQL query to get the daily stock data from database.
        sql = f'''
        select stock_code, date, volume, open_price, close_price, high_price, low_price, turnover, turnover_rate, price_change_percentage, amplitude 
        from stock_zh_a_hist_daily 
        where date between "{self.start_date}" and "{self.end_date}"
        '''
        data = MyEngine().read_sql_query(sql)
        return data 


if __name__ == '__main__':
    sc = StockChooser(start_date='2024-01-01', end_date='2024-08-20')
    sc.start()
    sc.data_candidates.to_csv(f'{sc.params.fig_dir}/candidates.csv', index=False)
    # sc.data_base.to_csv(f'{sc.params.fig_dir}/base.csv', index=False)
    # sc.analysis(['turnover_rate', 'amplitude'])

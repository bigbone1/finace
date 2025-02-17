import warnings
import backtrader as bt
import pandas as pd
import numpy as np
import akshare as ak
import pyfolio as pf
import matplotlib.pyplot as plt
# 忽略所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']

# 修改均线交叉判断（关键修正）
class OptimizedDualMA(bt.Strategy):
    params = (
        ('ma_short', 22),
        ('ma_long', 30),
        ('stop_loss', 0.02),
        ('take_profit', 1)
    )

    def __init__(self):
        self.ma_short = bt.indicators.SMA(self.data.close, period=self.params.ma_short)
        self.ma_long = bt.indicators.SMA(self.data.close, period=self.params.ma_long)
        self.crossover = bt.indicators.CrossOver(self.ma_short, self.ma_long)  # 新增交叉指标

    def next(self):
        if self.position:
            current_price = self.data.close[0]
            # 使用平均持仓价代替entry_price
            if current_price <= self.position.price * (1 - self.params.stop_loss):
                self.close()
            elif current_price >= self.position.price * (1 + self.params.take_profit):
                self.close()

        if not self.position and self.crossover > 0:  # 金叉信号
            self.order = self.buy()

        elif self.position and self.crossover < 0:     # 死叉信号
            self.order = self.sell()


if __name__ == '__main__':
    cerebro = bt.Cerebro(optreturn=False)

    # 添加数据
    dataframe = ak.stock_zh_a_hist(symbol="600078", period='daily', start_date='20240101')
    dataframe.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'}, inplace=True)
    dataframe['openinterest'] = 0
    dataframe.index = pd.to_datetime(dataframe['date'])
    data = bt.feeds.PandasData(dataname=dataframe)
    cerebro.adddata(data)

    # 策略参数优化
    # strats = cerebro.optstrategy(OptimizedDualMA)
    cerebro.addstrategy(OptimizedDualMA)

    # 回测设置
    cerebro.broker.setcash(1000000)
    cerebro.broker.setcommission(commission=0.0003)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)

    # 性能分析
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    results = cerebro.run()

    # 可视化
    cerebro.plot(style='candle')

    # 获取 PyFolio 分析器的结果
    strat = results[0]  # 获取第一个策略的第一个实例
    pyfoliozer = strat.analyzers.getbyname('pyfolio')
    returns, positions, transactions, gross_lev  = pyfoliozer.get_pf_items()

    # 使用 PyFolio 进行可视化
    pf.create_full_tear_sheet(
        returns,
        positions=positions,
        transactions=transactions,
        round_trips=True
    )
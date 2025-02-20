import numpy as np
import pandas as pd
import backtrader as bt
from backtrader.feeds import PandasData

# ======================== 数据预处理模块 ========================
class FactorData(PandasData):
    """
    自定义数据加载类（扩展PandasData以支持多因子）
    """
    lines = (
        'price_change_percentage', 'macd', 'kdj_k',
        'turnover_rate', 'ma_5', 'ma_20', 'hist_volatility_30d',
        'beta_252d', 'atr_14d'
    )
    params = (
        ('datetime', None),
        ('open', -1), ('high', -1), ('low', -1), 
        ('close', -1), ('volume', -1),
        ('price_change_percentage', -1),
        ('macd', -1),
        ('kdj_k', -1),
        ('turnover_rate', -1),
        ('ma_5', -1),
        ('ma_20', -1),
        ('hist_volatility_30d', -1),
        ('beta_252d', -1),
        ('atr_14d', -1),
    )

def load_stock_data(symbol, start_date='2020-01-01', end_date='2023-12-31'):
    """
    示例数据生成函数（实际使用时替换为真实数据）
    """
    dates = pd.date_range(start=start_date, end=end_date)
    n = len(dates)
    df = pd.DataFrame({
        'open': np.random.rand(n) * 100 + 100,
        'high': np.random.rand(n) * 105 + 100,
        'low': np.random.rand(n) * 95 + 100,
        'close': np.random.rand(n) * 100 + 100,
        'volume': np.random.randint(1e6, 1e7, size=n),
        'price_change_percentage': np.random.uniform(-0.1, 0.1, n),
        'macd': np.random.uniform(-2, 2, n),
        'kdj_k': np.random.uniform(0, 100, n),
        'turnover_rate': np.random.uniform(1, 10, n),
        'ma_5': np.random.rand(n) * 100 + 100,
        'ma_20': np.random.rand(n) * 100 + 100,
        'hist_volatility_30d': np.random.uniform(0.1, 0.5, n),
        'beta_252d': np.random.uniform(0.5, 1.5, n),
        'atr_14d': np.random.uniform(1, 5, n),
    }, index=dates)
    return df

# ======================== 策略逻辑模块 ========================
class MultiFactorStrategy(bt.Strategy):
    params = (
        ('top_n', 10),          # 持仓股票数量
        ('replace_n', 1),       # 每日替换数量
        ('score_threshold', 0), # 新股票得分阈值
        ('max_position_ratio', 0.15),  # 单股票最大仓位比例
        ('print_log', True),    # 是否打印交易日志
    )

    def __init__(self):
        self.stock_scores = {}  # 存储每只股票的实时得分
        self.order_dict = {}    # 跟踪订单
        self.trade_count = 0    # 交易计数器

    def log(self, txt, dt=None):
        """日志输出"""
        if self.params.print_log:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt}, {txt}')

    def prenext(self):
        self.next()

    def next(self):
        self.calculate_scores()
        self.rebalance_portfolio()
        self.log_positions()

    def calculate_scores(self):
        """计算每只股票的因子得分"""
        self.stock_scores.clear()
        for d in self.datas:
            # 获取指标值
            factors = {
                'price_change': d.price_change_percentage[0],
                'macd': d.macd[0],
                'kdj_k': d.kdj_k[0],
                'turnover': d.turnover_rate[0],
                'ma5_gt_ma20': 1 if d.ma_5[0] > d.ma_20[0] else -1,
                'volatility': d.hist_volatility_30d[0],
                'beta': d.beta_252d[0],
                'atr': d.atr_14d[0],
            }
            # 计算得分（示例公式）
            score = (
                0.25 * factors['price_change'] +
                0.20 * factors['macd'] +
                0.15 * factors['kdj_k'] +
                0.10 * factors['turnover'] +
                0.10 * factors['ma5_gt_ma20'] -
                0.10 * factors['volatility'] -
                0.05 * factors['beta'] -
                0.05 * factors['atr']
            )
            self.stock_scores[d._name] = score

    def rebalance_portfolio(self):
        """调仓逻辑"""
        # 当前持仓
        current_pos = {d: pos.size for d, pos in self.getpositions().items() if pos.size > 0}
        
        # 卖出逻辑：卖出得分最低的replace_n只
        sorted_stocks = sorted(self.stock_scores.items(), key=lambda x: x[1])
        sell_candidates = [stock for stock, _ in sorted_stocks[:self.p.replace_n] if stock in current_pos]
        
        for stock in sell_candidates:
            data = self.getdatabyname(stock)
            self.close(data)
            self.log(f'SELL {stock}, Price: {data.close[0]:.2f}')

        # 可用资金
        cash = self.broker.getcash()
        buy_power = cash / self.p.replace_n if self.p.replace_n > 0 else 0
        
        # 买入逻辑：选择非持仓中得分高于阈值的股票
        available_stocks = [s for s in self.stock_scores if s not in current_pos]
        buy_candidates = sorted(
            [(s, score) for s, score in self.stock_scores.items() 
             if s in available_stocks and score > self.p.score_threshold],
            key=lambda x: x[1], reverse=True
        )[:self.p.replace_n]

        for stock, score in buy_candidates:
            data = self.getdatabyname(stock)
            size = buy_power / data.close[0]
            if size * data.close[0] > self.broker.getvalue() * self.p.max_position_ratio:
                size = self.broker.getvalue() * self.p.max_position_ratio / data.close[0]
            self.buy(data=data, size=size)
            self.log(f'BUY {stock}, Price: {data.close[0]:.2f}, Size: {size:.2f}')

    def log_positions(self):
        """记录持仓"""
        pos = self.getpositions()
        pos_info = ', '.join([f'{d._name}: {p.size:.2f}' for d, p in pos.items()])
        self.log(f'持仓: {pos_info}')

# ======================== 回测执行模块 ========================
def run_backtest(data_dict, strategy_params=None):
    cerebro = bt.Cerebro(stdstats=False)
    
    # 添加数据
    for symbol, df in data_dict.items():
        data = FactorData(dataname=df, name=symbol)
        cerebro.adddata(data)
    
    # 添加策略
    cerebro.addstrategy(MultiFactorStrategy, **strategy_params or {})
    
    # 设置初始资金
    cerebro.broker.setcash(1_000_000)
    cerebro.broker.setcommission(commission=0.001)  # 0.1%佣金
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # 运行回测
    results = cerebro.run()
    return results[0]

# ======================== 参数优化模块 ========================
def optimize_strategy(data_dict, param_grid):
    cerebro = bt.Cerebro(optreturn=False)
    
    # 添加数据
    for symbol, df in data_dict.items():
        data = FactorData(dataname=df, name=symbol)
        cerebro.adddata(data)
    
    # 参数网格
    cerebro.optstrategy(
        MultiFactorStrategy,
        top_n=param_grid.get('top_n', [10]),
        replace_n=param_grid.get('replace_n', [1]),
        score_threshold=param_grid.get('score_threshold', [0]),
    )
    
    # 分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    # 运行优化
    opt_results = cerebro.run(maxcpus=1)
    
    # 提取最佳参数
    best_sharpe = -np.inf
    best_params = None
    for result in opt_results:
        sharpe = result.analyzers.sharpe.get_analysis()['sharperatio']
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = result.params._getkwargs()
    return best_params, best_sharpe


if __name__ == '__main__':
    # 1. 生成示例数据（10只股票）
    symbols = [f'Stock_{i}' for i in range(1, 11)]
    data_dict = {s: load_stock_data(s) for s in symbols}
    
    # 2. 参数优化
    param_grid = {
        'top_n': [8, 10, 12],
        'replace_n': [1, 2],
        'score_threshold': [-0.5, 0, 0.5],
    }
    best_params, best_sharpe = optimize_strategy(data_dict, param_grid)
    print(f'最佳参数: {best_params}, 夏普比率: {best_sharpe:.2f}')
    
    # 3. 使用最佳参数运行回测
    result = run_backtest(data_dict, best_params)
    
    # 4. 输出回测结果
    print('\n========== 回测结果 ==========')
    print(f"夏普比率: {result.analyzers.sharpe.get_analysis()['sharperatio']:.2f}")
    print(f"最大回撤: {result.analyzers.drawdown.get_analysis()['max']['drawdown']:.2f}%")
    print(f"年化收益: {result.analyzers.returns.get_analysis()['rnorm100']:.2f}%")
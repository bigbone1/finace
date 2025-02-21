from database.factor_calculator import FullFactorCalculator
from backtrader.feeds import PandasData  # 新增导入
import backtrader as bt
import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)

class EnhancedFactorData(PandasData):
    """增强版数据类支持全量因子"""
    lines = (
        # 基础行情数据
        'open', 'high', 'low', 'close', 'volume',
        # 技术指标
        'ma_5', 'ma_20', 'macd', 'kdj_k', 'rsi',
        # 估值因子
        'pe', 'pb', 
        # 成长因子
        'net_profit_growth',
        # 风险指标
        'hist_volatility_30d', 'beta_252d'
    )
    
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        ('ma_5', 'ma_5'),
        ('ma_20', 'ma_20'),
        ('macd', 'macd'),
        ('kdj_k', 'kdj_k'),
        ('rsi', 'rsi'),
        ('pe', 'pe'),
        ('pb', 'pb'),
        ('net_profit_growth', 'net_profit_growth'),
        ('hist_volatility_30d', 'hist_volatility_30d'),
        ('beta_252d', 'beta_252d')
    )

# ======================== 策略逻辑模块 ========================
class OptimizedMultiFactorStrategy(bt.Strategy):
    params = (
        ('position_size', 10),    # 持仓数量
        ('max_daily_change', 2),  # 每日最大调仓数
        ('position_ratio', 0.15), # 单股最大仓位比例
        ('printlog', False)       # 是否打印交易日志
    )

    def __init__(self):
        self.initial_portfolio = []  # 初始股票池
        self.current_holdings = {}   # 当前持仓
        self.trade_counter = 0       # 交易计数器
        
        # 初始化阶段计算全市场评分
        if len(self.datas) == 1:  # 仅运行在第一个数据
            all_scores = self.calculate_initial_scores()
            self.initial_portfolio = sorted(
                all_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:self.params.position_size]

    def calculate_initial_scores(self):
        """初始化阶段全市场评分"""
        scores = {}
        for d in self.datas:
            if len(d.close) > 20:  # 确保足够数据长度
                try:
                    scores[d._name] = self.factor_score(d)
                except Exception as e:
                    self.log(f"评分错误 {d._name}: {str(e)}")
        return scores

    def factor_score(self, data):
        """多因子综合评分"""
        # 估值因子（越低越好）
        value_score = 0.4 * (1 / data.pe[0]) + 0.6 * (1 / data.pb[0])
        
        # 成长因子（越高越好）
        growth_score = 0.7 * data.net_profit_growth[0] 
        
        # 技术指标（动态权重）
        tech_score = (
            0.3 * data.rsi[0]/100 +
            0.4 * (data.close[0] > data.ma_20[0]) +
            0.3 * data.macd[0]
        )
        
        # 风险调整（越低越好）
        risk_penalty = 0.6 * data.hist_volatility_30d[0] + 0.4 * abs(data.beta_252d[0]-1)
        
        return (value_score * 0.35 + 
                growth_score * 0.3 + 
                tech_score * 0.25 - 
                risk_penalty * 0.1)

    def next(self):
        # 初始化阶段不交易
        if len(self.datas[0]) < 2:
            return
            
        # 每日调仓逻辑
        self.rebalance_portfolio()

    def rebalance_portfolio(self):
        current_pos = {d._name: d for d in self.getpositions()}
        candidate_pool = self.get_candidates(current_pos)
        
        # 卖出逻辑
        sell_list = self.get_weakest_positions(current_pos)
        for symbol in sell_list[:self.params.max_daily_change]:
            self.close(current_pos[symbol])
            self.log(f'卖出 {symbol}')
            
        # 买入逻辑
        buy_power = self.broker.getcash() / self.params.max_daily_change
        for symbol in candidate_pool[:self.params.max_daily_change]:
            data = self.getdatabyname(symbol)
            size = min(buy_power/data.close[0], 
                      self.broker.getvalue()*self.params.position_ratio/data.close[0])
            self.buy(data, size=size)
            self.log(f'买入 {symbol} {size:.2f}股')

    def get_candidates(self, current_pos):
        """获取候选股票列表"""
        scores = {}
        for d in self.datas:
            if d._name not in current_pos and len(d) > 20:
                scores[d._name] = self.factor_score(d)
        return sorted(scores.keys(), 
                     key=lambda x: scores[x], 
                     reverse=True)

    def get_weakest_positions(self, current_pos):
        """获取持仓中最差的股票"""
        scores = {}
        for symbol, pos in current_pos.items():
            data = self.getdatabyname(symbol)
            scores[symbol] = self.factor_score(data)
        return sorted(scores.keys(), key=lambda x: scores[x])

    def log(self, txt, dt=None):
        """日志记录"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}, {txt}')

    def stop(self):
        print(f"总交易次数: {len(self._orders)}")
        print(f"最终持仓数量: {len(self.getpositions())}")
        # 在策略结束时验证交易记录
        if len(self._orders) == 0:
            print("警告：本次回测未产生任何交易！")
            print("可能原因：")
            print("- 初始评分未选出有效股票")
            print("- 调仓条件未触发")
            print("- 可用资金不足")

# ======================== 数据加载模块 ========================        
def load_factor_data(symbol, engine, start_date, end_date):
    """从数据库加载多维度因子数据"""
    query = f"""
    SELECT 
        h.trade_date,
        h.open, h.high, h.low, h.close, h.volume,
        h.ma_5, h.ma_20, h.macd, h.kdj_k, h.rsi,
        v.pe, v.pb,
        g.net_profit_growth,
        h.hist_volatility_30d,
        h.beta_252d
    FROM stock_zh_a_hist h
    LEFT JOIN valuation_factors v 
        ON h.symbol = v.symbol AND h.trade_date = v.trade_date
    LEFT JOIN (
        SELECT 
            symbol,
            announcement_date as last_announce,
            net_profit_growth,
            ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY announcement_date DESC) as rn
        FROM growth_factors
    ) g ON h.symbol = g.symbol 
        AND h.trade_date >= g.last_announce
        AND g.rn = 1  -- 取最新公告记录
    WHERE h.symbol = '{symbol}'
        AND h.trade_date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY h.trade_date
    """
    df = pd.read_sql(query, engine)
    
    # ========== 新增关键处理 ==========
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index().asfreq('D')  # 确保每日频率
    

    # 标准化方法需要离线统一计算，以便在线预测时与回测时使用同样的标准化方法
    # 处理缺失值
    df['net_profit_growth'] = df['net_profit_growth'].ffill()
    df['net_profit_growth'] = (
    df['net_profit_growth']
    .astype('float32')  # 明确指定数据类型
    .ffill()
    .fillna(0.5)
)

    # 数据标准化
    numeric_cols = ['pe', 'pb', 'net_profit_growth', 'rsi', 'macd']
    df[numeric_cols] = df[numeric_cols].apply(
        lambda x: (x - x.mean()) / x.std()
    )
    if 'beta_252d' in df.columns:
        # 先处理其他字段的缺失
        other_cols = df.columns.difference(['beta_252d', 'net_profit_growth'])
        df = df.dropna(subset=other_cols)
        
        # 对beta_252d进行填充
        df['beta_252d'] = (
            df['beta_252d']
            .astype('float32')
            .fillna(0.5)
            .clip(lower=-10, upper=10)  # 限制合理范围
            .replace([np.inf, -np.inf], 0.5)
        )
    else:
        df['beta_252d'] = 0.5  # 容错处理
    return df

# ======================== 回测执行模块 ========================
def run_enhanced_backtest(symbol_list, start_date, end_date):
    cerebro = bt.Cerebro(stdstats=False)
    
    # 初始化因子计算器
    calculator = FullFactorCalculator()
    
    # 添加数据
    for symbol in symbol_list:
        df = load_factor_data(symbol, calculator.engine, start_date, end_date)
        
        if not df.empty:
            print(f"加载数据 {symbol}, {df.shape}")
            data = EnhancedFactorData(dataname=df, name=symbol)
            cerebro.adddata(data)
    
    # 策略参数
    cerebro.addstrategy(OptimizedMultiFactorStrategy)
    
    # 回测配置
    cerebro.broker.setcash(1_000_000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=95)
    
    # 分析器
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    
    # 运行回测
    results = cerebro.run()
    return results[0]

# ======================== 结果分析模块 ========================
def analyze_results(result):
    """专业级结果分析"""
    pyfoliozer = result.analyzers.getbyname('pyfolio')
    # 修改为接收4个返回值
    returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
    
    sharpe_analyzer = result.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe_analyzer.get('sharperatio', None)
    if sharpe_ratio is None:
        print("夏普比率计算失败，可能原因：")
        print("- 回测时间不足1年")
        print("- 未产生任何交易")
        print("- 所有收益率为零")
        sharpe_ratio = 0.0  # 设置默认值
    
    print("\n========== 高级分析报告 ==========")
    print(f"夏普比率: {sharpe_ratio:.2f}" if sharpe_ratio is not None else "夏普比率: 无效")
    print(f"杠杆率数据样例：{gross_lev[:5]}")

    # 生成可视化报告（增加杠杆率参数）
    import pyfolio as pf
    pf.create_full_tear_sheet(
        returns,
        positions=positions,
        transactions=transactions,
        live_start_date='2023-06-01'
    )
        # 可选：单独分析杠杆率
    if not gross_lev.empty:
        print("\n杠杆率分析:")
        print(gross_lev.describe())

if __name__ == '__main__':
    # 初始化因子数据库
    factor_calculator = FullFactorCalculator()
    
    # 获取全市场股票
    all_symbols = factor_calculator.get_all_stocks()[:270]  # 前200只测试
    
    # 运行增强版回测
    result = run_enhanced_backtest(
        symbol_list=all_symbols,
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # 生成分析报告
    analyze_results(result)
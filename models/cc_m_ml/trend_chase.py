from libs import MyEngine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号"-"显示为方块的问题

def get_all_stock_codes():
    """获取所有股票代码"""
    sql = 'SELECT DISTINCT code FROM stock_zh_5_minute'  # 去重获取所有股票代码
    codes_df = MyEngine().read_sql_query(sql)
    return codes_df['code'].tolist()

# ================== 数据加载 ==================
def load_data(code):
    """加载并预处理数据"""
    start_date = '2024-01-01'
    end_date = '2024-08-20'
    table = 'stock_zh_5_minute'
    cols = ['date', 'time', 'code', 'open', 'close', 'high', 'low', 'close', 'volume', 'amount']
    # sql = MyEngine().create_sql(cols, table_name=table, 
    #                             start_date=start_date, end_date=end_date, 
    #                             codes=['sh.600078'])
    sql = rf'select * from stock_zh_5_minute where code in ("{code}")'
    df = MyEngine().read_sql_query(sql)
    df.sort_values(by=['code', 'date', 'time'], inplace=True)  # 按时间排序
    df.set_index('date', inplace=True)
    df = df.astype(float, errors='ignore')
    return df

# ================== 策略逻辑 ==================
def calculate_ma_signals(df, short_window=5, long_window=20):
    """计算双均线信号"""
    # 计算双均线
    df['ma_long'] = df['close'].rolling(window=long_window).mean()
    df['ma_short'] = df['close'].rolling(window=short_window).mean()
    df['signal'] = 0  # 0: 空仓, 1: 持仓
    df['signal'] = np.where(df['ma_short'] > df['ma_long'], 1, 0)
    df['position'] = df['signal'].diff()  # 持仓变化：1买入，-1卖出
    return df

# ================== 回测引擎 ==================
def backtest_strategy(df, commission_rate=0.0003, initial_cash=1000000):
    """执行回测计算"""
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    
    # 计算交易费用
    df.reset_index(drop=True, inplace=True)
    trade_positions = df[df['position'] != 0].index
    df['commission'] = 0
    df.loc[trade_positions, 'commission'] = abs(df['position'] *df['close']* commission_rate)
    
    # 累计收益计算
    df['net_returns'] = df['strategy_returns'] - df['commission']
    df['cumulative_returns'] = (1 + df['net_returns']).cumprod()
    df['cumulative_benchmark'] = (1 + df['returns']).cumprod()
    
    return df

# ================== 绩效评估 ==================
def evaluate_performance(df):
    """计算关键绩效指标"""
    total_return = df['cumulative_returns'].iloc[-1] - 1
    annual_return = total_return ** (252/len(df)) - 1  # 按年化计算
    max_drawdown = (df['cumulative_returns'].cummax() - df['cumulative_returns']).max()
    sharpe_ratio = df['net_returns'].mean() / df['net_returns'].std() * np.sqrt(252)
    
    print(f"累计收益率：{total_return:.2%}")
    print(f"年化收益率：{annual_return:.2%}")
    print(f"最大回撤：{max_drawdown:.2%}")
    print(f"夏普比率：{sharpe_ratio:.2f}")

# ================== 可视化 ==================
def plot_results(df, stock_code):
    """绘制带完整信息的收益曲线"""
    plt.figure(figsize=(14,7))
    plt.plot(df['cumulative_returns'], label='策略收益', linewidth=2, color='#2ca02c')
    plt.plot(df['cumulative_benchmark'], label='基准收益', linestyle='--', color='#1f77b4')
    
    # 图表元素增强
    plt.title(f'{stock_code}双均线策略收益对比\n{df.index.min()} - {df.index.max()}', 
             fontsize=14, pad=20)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('累计收益率', fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 添加关键数据标注
    max_return = df['cumulative_returns'].max()
    min_return = df['cumulative_returns'].min()
    plt.annotate(f'峰值收益：{max_return:.2f}', 
                xy=(df['cumulative_returns'].idxmax(), max_return),
                xytext=(20, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->'))
    
    # 保存图表
    plt.savefig(f'results/{stock_code}_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


# ================== 主程序 ==================
if __name__ == "__main__":
    all_codes = get_all_stock_codes()
    performance_df = pd.DataFrame()
    for code in all_codes:
        try:
            data = load_data(code)  # 修改load_data函数接收code参数
            data = calculate_ma_signals(data)
            data = backtest_strategy(data)
            
            # 存储绩效指标
            metrics = {
                'code': code,
                '累计收益率': data['cumulative_returns'].iloc[-1] - 1,
                '年化收益率': (data['cumulative_returns'].iloc[-1]**(252/len(data)))-1,
                '最大回撤': (data['cumulative_returns'].cummax()-data['cumulative_returns']).max(),
                '夏普比率': data['net_returns'].mean() / data['net_returns'].std() * np.sqrt(252)
            }
            performance_df = pd.concat([performance_df, pd.DataFrame([metrics])], ignore_index=True)
            
            # 生成并保存图表
            plot_results(data, code)
            
        except Exception as e:
            print(f"股票{code}处理失败：{str(e)}")
    
    # 保存所有绩效结果
    performance_df.to_csv('strategy_performance.csv', index=False)


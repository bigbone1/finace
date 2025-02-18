from concurrent.futures import ThreadPoolExecutor
import akshare as ak
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FullFactorCalculator:
    def __init__(self):
        username = 'root'
        password = ''
        host = '127.0.0.1:3306'
        database = 'finance'
        db_url = f'mysql+pymysql://{username}:{password}@{host}/{database}'
        self.engine = create_engine(db_url)
        self.index_symbol = 'sh000001'  # 上证指数作为市场基准
        
    # 辅助方法 --------------------------------------------------
    def get_all_stocks(self):
        """获取全A股代码列表（带交易所前缀）"""
        spot_df = ak.stock_zh_a_spot_em()
        spot_df['symbol'] = np.where(spot_df['代码'].str.startswith('6'), 
                                   'sh' + spot_df['代码'], 
                                   'sz' + spot_df['代码'])
        return spot_df['symbol'].tolist()

    def get_hist_data(self, symbol):
        """获取复权历史行情"""
        try:
            return ak.stock_zh_a_daily(symbol=symbol, adjust="hfq")
        except Exception as e:
            logging.error(f"Failed to get historical data for {symbol}: {e}")
            return None
    
    # 估值因子 --------------------------------------------------
    def update_valuation_factors(self, symbol):
        """更新估值因子数据"""
        try:
            df_indicator = ak.stock_a_indicator_lg(symbol=symbol[2:])
            if df_indicator.empty:
                logging.warning(f"No valuation data found for {symbol}")
                return
            df_indicator['symbol'] = symbol
            df_indicator['trade_date'] = pd.to_datetime(df_indicator['trade_date'])
            df_indicator[['symbol', 'trade_date', 'pe', 'pe_ttm', 'pb', 'dv_ratio', 'dv_ttm', 'ps', 'ps_ttm', 'total_mv']].to_sql(
                'valuation_factors', self.engine, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"Valuation update failed for {symbol}: {str(e)}")

    # 成长因子 --------------------------------------------------
    def calculate_growth_factors(self, symbol):
        """计算成长因子"""
        try:
            # 获取数据并检查空值
            income_df = ak.stock_financial_report_sina(stock=symbol, symbol="利润表")
            cash_df = income_df.copy()

            if income_df.empty or cash_df.empty:
                logging.warning(f"No financial report data found for {symbol}")
                return

            # 按时间排序并清洗数据
            income_df = income_df.sort_values("报告日").dropna(subset=["净利润", "营业收入"])
            cash_df = cash_df.sort_values("报告日").dropna(subset=["研发费用"])

            # 计算净利润同比增长率（修复分母为 None 的问题）-----------------------------
            net_profit = income_df[["报告日", "净利润"]].copy()
            if len(net_profit) < 5:  # 至少需要5个季度数据才能计算同比
                logging.warning(f"{symbol} 净利润数据不足5个季度，无法计算同比增长率")
                return

            net_profit["net_profit_shift4"] = net_profit["净利润"].shift(4)
            # 过滤掉 shift4 为空或零的行
            net_profit = net_profit.dropna(subset=["net_profit_shift4"])
            net_profit = net_profit[net_profit["net_profit_shift4"] != 0]

            if net_profit.empty:
                logging.warning(f"{symbol} 净利润数据不足或去年同期数据无效")
                return

            # 安全计算增长率（避免除零和 None）
            net_profit["net_profit_growth"] = (
                (net_profit["净利润"] - net_profit["net_profit_shift4"]) 
                / net_profit["net_profit_shift4"].abs() 
                * 100
            )

            # 计算营业收入同比增长率（同理）------------------------------------------
            revenue = income_df[["报告日", "营业收入"]].copy()
            if len(revenue) < 5:
                logging.warning(f"{symbol} 营业收入数据不足5个季度，无法计算同比增长率")
                return

            revenue["revenue_shift4"] = revenue["营业收入"].shift(4)
            revenue = revenue.dropna(subset=["revenue_shift4"])
            revenue = revenue[revenue["revenue_shift4"] != 0]

            if revenue.empty:
                logging.warning(f"{symbol} 营业收入数据不足或去年同期数据无效")
                return

            revenue["revenue_growth"] = (
                (revenue["营业收入"] - revenue["revenue_shift4"]) 
                / revenue["revenue_shift4"].abs() 
                * 100
            )

            # 计算研发费用环比增长率（修复分母为 None 的问题）-------------------------
            rd_expense = cash_df[["报告日", "研发费用"]].copy()
            if len(rd_expense) < 2:  # 至少需要2个季度计算环比
                logging.warning(f"{symbol} 研发费用数据不足2个季度，无法计算环比增长率")
                return

            rd_expense["rd_expense_shift1"] = rd_expense["研发费用"].shift(1)
            rd_expense = rd_expense.dropna(subset=["rd_expense_shift1"])
            rd_expense = rd_expense[rd_expense["rd_expense_shift1"] != 0]

            if rd_expense.empty:
                logging.warning(f"{symbol} 研发费用数据不足或上季度数据无效")
                return

            rd_expense["rd_expense_growth"] = (
                (rd_expense["研发费用"] - rd_expense["rd_expense_shift1"]) 
                / rd_expense["rd_expense_shift1"].abs() 
                * 100
            )

            # 合并数据并保存结果（同之前代码）
            merged_df = pd.merge(net_profit, revenue, on="报告日", how="inner")
            merged_df = pd.merge(merged_df, rd_expense, on="报告日", how="inner")
            merged_df["symbol"] = symbol
            merged_df.rename(columns={"报告日": "announcement_date"}, inplace=True)

            output_cols = ["symbol", "announcement_date", "net_profit_growth", "revenue_growth", "rd_expense_growth"]
            merged_df[output_cols].to_sql(
                "growth_factors", 
                self.engine, 
                if_exists="append", 
                index=False
            )
        except Exception as e:
            logging.error(f"Growth factor error for {symbol}: {str(e)}")

    # 波动率因子 --------------------------------------------------
    def calculate_volatility(self, symbol, window=30):
        """计算波动率相关因子"""
        try:
            stock_data = self.get_hist_data(symbol)
            index_data = self.get_hist_data(self.index_symbol)
            
            if stock_data is None or index_data is None:
                logging.warning(f"No historical data found for {symbol} or index")
                return
            
            if 'date' not in stock_data.columns or 'date' not in index_data.columns:
                logging.warning(f"Missing 'date' column in historical data for {symbol} or index")
                return
            
            stock_data['log_return'] = np.log(stock_data['close'] / stock_data['close'].shift(1))
            index_data['log_return'] = np.log(index_data['close'] / index_data['close'].shift(1))
            
            stock_data['hist_vol'] = stock_data['log_return'].rolling(window).std() * np.sqrt(252)
            stock_data['hist_vol_5'] = stock_data['log_return'].rolling(5).std() * np.sqrt(252)
            
            merged_returns = pd.merge(stock_data[['date', 'log_return']], 
                                    index_data[['date', 'log_return']], 
                                    on='date', suffixes=('_stock', '_index'))
            cov_matrix = merged_returns[['log_return_stock', 'log_return_index']].rolling(252).cov()
            beta = cov_matrix.xs('log_return_stock', level=1)['log_return_index'] / \
                 merged_returns['log_return_index'].rolling(252).var()
            
            high_low = stock_data['high'] - stock_data['low']
            high_close = np.abs(stock_data['high'] - stock_data['close'].shift())
            low_close = np.abs(stock_data['low'] - stock_data['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            result_df = pd.DataFrame({
                'trade_date': stock_data['date'],
                'hist_volatility_30d': stock_data['hist_vol'],
                'hist_volatility_5d': stock_data['hist_vol_5'],
                'beta_252d': beta.values,
                'atr_14d': atr
            })
            result_df['symbol'] = symbol
            result_df[['symbol', 'trade_date', 'hist_volatility_30d', 'hist_volatility_5d',
                      'beta_252d', 'atr_14d']].to_sql(
                'volatility_factors', self.engine, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"Volatility calculation error for {symbol}: {str(e)}")

    # 质量因子 --------------------------------------------------
    def calculate_quality_factors(self, symbol):
        """计算质量因子"""
        try:
            balance_df = ak.stock_financial_report_sina(stock=symbol, symbol="资产负债表")
            income_df = ak.stock_financial_report_sina(stock=symbol, symbol="利润表")
            cash_df = ak.stock_financial_report_sina(stock=symbol, symbol="现金流量表")
            
            if balance_df is None or income_df is None or cash_df is None:
                logging.warning(f"No financial report data found for {symbol}")
                return
            
            debt_ratio = balance_df[balance_df['报表类型'] == '年报'][['公告日期', '负债率']]
            if debt_ratio.empty:
                logging.warning(f"No debt ratio data found for {symbol}")
                return
            
            debt_ratio['debt_to_asset'] = debt_ratio['负债率'].str.replace('%', '').astype(float)
            
            cash_flow = cash_df[cash_df['报表类型'] == '年报'][['公告日期', '经营活动现金流净额']]
            net_profit = income_df[income_df['报表类型'] == '年报'][['公告日期', '净利润']]
            merged = pd.merge(cash_flow, net_profit, on='公告日期')
            merged['cash_flow_ratio'] = merged['经营活动现金流净额'].astype(float) / \
                                      merged['净利润'].astype(float)
            
            merged['accruals'] = merged['净利润'].astype(float) - \
                               merged['经营活动现金流净额'].astype(float)
            merged['accruals_ratio'] = merged['accruals'] / \
                                     merged['净利润'].astype(float)
            
            final_df = pd.merge(debt_ratio, merged, on='公告日期')
            final_df['symbol'] = symbol
            final_df[['symbol', '公告日期', 'debt_to_asset', 
                     'cash_flow_ratio', 'accruals_ratio']].to_sql(
                'quality_factors', self.engine, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"Quality factor error for {symbol}: {str(e)}")

    # 技术指标 --------------------------------------------------
    def calculate_technical_indicators(self, symbol):
        """计算技术指标"""
        try:
            df = self.get_hist_data(symbol)
            if df is None or 'date' not in df.columns:
                logging.warning(f"No historical data or missing 'date' column for {symbol}")
                return
            
            df['ma_5'] = df['close'].rolling(5).mean()
            df['ma_20'] = df['close'].rolling(20).mean()
            
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            df['macd'] = macd_line - signal_line
            
            low_min = df['low'].rolling(9).min()
            high_max = df['high'].rolling(9).max()
            rsv = (df['close'] - low_min) / (high_max - low_min) * 100
            df['kdj_k'] = rsv.ewm(com=2).mean()
            
            df['symbol'] = symbol
            df[['symbol', 'date', 'ma_5', 'ma_20', 'macd', 'kdj_k']].to_sql(
                'technical_indicators', self.engine, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"Technical indicator error for {symbol}: {str(e)}")

    def initialize_hist_data(self):
        """初始化全量历史数据"""
        # 创建表
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS valuation_factors (
            symbol VARCHAR(10), -- 证券代码
            trade_date DATE, -- 交易日期
            pe FLOAT, -- 市盈率
            pe_ttm FLOAT, -- 市盈率（TTM）
            pb FLOAT, -- 市净率
            dv_ratio FLOAT, -- 股息率
            dv_ttm FLOAT, -- 股息率（TTM）
            ps FLOAT, -- 市销率
            ps_ttm FLOAT, -- 市销率（TTM）
            total_mv FLOAT -- 总市值
        );
        """
        with self.engine.connect() as connection:
            connection.execute(text(create_table_sql))
        
        stocks = self.get_all_stocks()
        
        for symbol in stocks:
            self._init_single_stock(symbol)
            break
        
        self._init_macro_data()
        self._init_margin_data()

    def _init_single_stock(self, symbol):
        """单只股票历史数据初始化"""
        logging.info(f"Processing {symbol}")
        try:
            self.update_valuation_factors(symbol)
            self.calculate_growth_factors(symbol)
            self.calculate_volatility(symbol)
            self.calculate_quality_factors(symbol)
            self.calculate_technical_indicators(symbol)
        except Exception as e:
            logging.error(f"Initialize failed for {symbol}: {str(e)}")

    def _init_macro_data(self):
        """初始化宏观经济数据"""
        macro_functions = {
            'gdp': ak.macro_china_gdp,
            'cpi': ak.macro_china_cpi,
            'ppi': ak.macro_china_ppi,
            'm2': ak.macro_china_m2_yearly
        }
        
        for name, func in macro_functions.items():
            df = func()
            df['indicator'] = name
            df.to_sql('macro_data', self.engine, if_exists='append', index=False)

    def _init_margin_data(self):
        """初始化融资融券历史数据"""
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')
        
        sse_df = ak.stock_margin_sse(start_date=start_date)
        szse_df = ak.stock_margin_szse(start_date=start_date)
        
        margin_df = pd.concat([sse_df, szse_df])
        margin_df.to_sql('margin_data', self.engine, if_exists='append', index=False)
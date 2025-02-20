from concurrent.futures import ThreadPoolExecutor
import akshare as ak
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import logging
import time 

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
        self.index_data = self.get_hist_data(self.index_symbol, type='index')
        
    def daily_update(self, start_date=pd.to_datetime("today").date()):
        """每日更新所有股票的最近数据"""
        logging.info("Starting daily update of stock factors")
        stocks = self.get_all_stocks()
        self.daily_start_date = start_date

        for symbol in stocks:
            if 'unknown' in symbol:
                continue
            try:
                # 获取最近的历史数据, 为保证beta252天数据完整性
                start_date = start_date - pd.DateOffset(months=13)
                self.symbol_hist_data = self.get_hist_data(symbol, start_date)
                if self.symbol_hist_data is None or self.symbol_hist_data.empty:
                    logging.warning(f"No historical data found for {symbol}")
                    continue

                # 仅处理最近的数据
                latest_data = self.symbol_hist_data.iloc[-1:]

                # 更新估值因子
                self.update_valuation_factors(symbol)

                # 计算技术指标
                self.calculate_technical_indicators(symbol)

                # TODO: 迁移质量因子-成长因子到对应的更新周期函数
                # 计算成长因子
                # self.calculate_growth_factors(symbol)

                # 计算质量因子
                # self.calculate_quality_factors(symbol)
            except Exception as e:
                logging.error(f"Daily update failed for {symbol}: {str(e)}")
        logging.info("Daily update completed")
    def clear(self):
        # 清除全部表的数据
        with self.engine.connect() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
            conn.execute(text("TRUNCATE TABLE valuation_factors"))
            conn.execute(text("TRUNCATE TABLE growth_factors"))
            conn.execute(text("TRUNCATE TABLE quality_factors"))   
            conn.execute(text("TRUNCATE TABLE stock_zh_a_hist")) 
            conn.execute(text("TRUNCATE TABLE macro_data")) 
            conn.execute(text("TRUNCATE TABLE margin_data")) 
            conn.execute(text("TRUNCATE TABLE sector_indices"))  # 新增：清空行业指数表

    # 辅助方法 --------------------------------------------------
    def get_all_stocks(self):
        """获取全A股代码列表（带交易所前缀）"""
        spot_df = ak.stock_zh_a_spot_em()
        
        # 添加对空值或异常值的检查
        if spot_df is None or spot_df.empty:
            logging.error("Failed to fetch stock data from akshare")
            return []

        # 清洗数据，去除无效行
        spot_df = spot_df.dropna(subset=['代码'])

        # 处理不同类型的股票代码
        def format_symbol(code):
            if isinstance(code, str) and len(code) == 6:
                if code.startswith('6'):
                    return f'sh{code}'
                elif code.startswith(('0', '3')):
                    return f'sz{code}'
                elif code.startswith(('8', '4')):  # 北交所代码
                    return f'bj{code}'
                else:
                    logging.warning(f"Unknown symbol format: {code}")
                    return f'unknown_{code}'
            else:
                logging.warning(f"Invalid code format: {code}")
                return f'invalid_{code}'

        spot_df['symbol'] = spot_df['代码'].apply(format_symbol)
        return spot_df['symbol'].tolist()

    def get_hist_data(self, symbol, start_date=None, type='stock'):
        """获取复权历史行情"""
        try: 
            if type != 'stock':
                return ak.stock_zh_index_daily(symbol)
            else:
                return ak.stock_zh_a_hist(symbol=symbol[2:], adjust="qfq", start_date=start_date)
        except Exception as e:
            logging.error(f"Failed to get historical data for {symbol}: {e}")
            return None

    # 新增：获取行业指数数据
    def get_sector_indices(self):
        """获取行业指数数据"""
        try:
            sector_indices = ak.stock_sector_summary()
            sector_indices['trade_date'] = pd.to_datetime(sector_indices['日期'])
            sector_indices.rename(columns={'代码': 'index_code', '名称': 'name', '最新价': 'close', '涨跌幅': 'change_rate'}, inplace=True)
            sector_indices['open'] = np.nan  # 行业指数数据中可能没有开盘价
            sector_indices['high'] = np.nan  # 行业指数数据中可能没有最高价
            sector_indices['low'] = np.nan  # 行业指数数据中可能没有最低价
            sector_indices['volume'] = np.nan  # 行业指数数据中可能没有成交量
            sector_indices['turnover'] = np.nan  # 行业指数数据中可能没有成交额
            return sector_indices[['index_code', 'trade_date', 'open', 'close', 'high', 'low', 'volume', 'turnover']]
        except Exception as e:
            logging.error(f"Failed to get sector indices data: {e}")
            return None

    # 新增：更新行业指数数据
    def update_sector_indices(self):
        """更新行业指数数据"""
        try:
            sector_indices_df = self.get_sector_indices()
            if sector_indices_df is None or sector_indices_df.empty:
                logging.warning("No sector indices data found")
                return
            sector_indices_df.to_sql('sector_indices', self.engine, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"Sector indices update failed: {str(e)}")

    # 估值因子 --------------------------------------------------
    def update_valuation_factors(self, symbol, update_type='all'):
        """更新估值因子数据"""
        try:
            df_indicator = ak.stock_a_indicator_lg(symbol=symbol[2:])
            if df_indicator.empty:
                logging.warning(f"No valuation data found for {symbol}")
                return
            if update_type == 'all':
                pass 
            elif update_type == 'daily':
                start_date = pd.to_datetime(self.daily_start_date).date()
                end_date = pd.to_datetime('today').date()
                df_indicator = df_indicator[df_indicator['trade_date'].between(start_date, end_date)]
            else:
                raise ValueError("Invalid update_type")
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
            stock_data = self.symbol_hist_data.copy()
            index_data = self.index_data 
            stock_data.rename(columns={'日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高':'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'turnover', 
                    '振幅': 'amplitude',
                    '涨跌幅': 'price_change_percentage',
                    '涨跌额': 'price_change_amount',
                    '换手率': 'turnover_rate'}, inplace=True)
            if stock_data is None or index_data is None:
                logging.warning(f"No historical data found for {symbol} or index")
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
            return result_df[['trade_date', 'hist_volatility_30d', 'hist_volatility_5d',
                      'beta_252d', 'atr_14d']]
            
        except Exception as e:
            logging.error(f"stock_zh_a_hist error for {symbol}: {str(e)}")
            return None

    # 质量因子 --------------------------------------------------
    def calculate_quality_factors(self, symbol):
        """计算质量因子"""
        try:
            balance_df = ak.stock_financial_report_sina(stock=symbol, symbol="资产负债表")
            time.sleep(30)
            income_df = ak.stock_financial_report_sina(stock=symbol, symbol="利润表")
            time.sleep(30)
            cash_df = ak.stock_financial_report_sina(stock=symbol, symbol="现金流量表")
            
            if balance_df is None or income_df is None or cash_df is None:
                logging.warning(f"No financial report data found for {symbol}")
                return
            
            debt_ratio = balance_df[['报告日', '负债合计', '资产总计']]
            if debt_ratio.empty:
                logging.warning(f"No debt ratio data found for {symbol}")
                return
            debt_ratio.loc[:, 'debt_to_asset'] = (debt_ratio['负债合计'] / debt_ratio['资产总计']) * 100
            
            cash_flow = cash_df[['报告日', '经营活动产生的现金流量净额']]
            net_profit = income_df[['报告日', '净利润']]
            merged = pd.merge(cash_flow, net_profit, on='报告日')
            merged['cash_flow_ratio'] = merged['经营活动产生的现金流量净额'].astype(float) / merged['净利润'].astype(float)
            
            merged['accruals'] = merged['净利润'].astype(float) - \
                               merged['经营活动产生的现金流量净额'].astype(float)
            merged['accruals_ratio'] = merged['accruals'] / \
                                     merged['净利润'].astype(float)
            
            final_df = pd.merge(debt_ratio, merged, on='报告日')
            final_df['symbol'] = symbol
            final_df.rename(columns={'报告日': 'announcement_date'}, inplace=True)
            final_df[['symbol', 'announcement_date', 'debt_to_asset', 
                     'cash_flow_ratio', 'accruals_ratio']].to_sql(
                'quality_factors', self.engine, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"Quality factor error for {symbol}: {str(e)}")
    
    def calculate_rsi(self, df, window=14):
        """计算相对强弱指数（RSI）"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_obv(self, df):
        """计算累积/派发量（OBV）"""
        df['direction'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
        df['volume_signed'] = df['direction'] * df['volume']
        obv = df['volume_signed'].cumsum()
        return obv

    def calculate_vwap(self, df):
        """计算成交量加权平均价格（VWAP）"""
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        return df['vwap']

    def calculate_cci(self, df, window=20):
        """计算商品通道指数（CCI）"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        moving_average = typical_price.rolling(window=window).mean()
        def custom_std(x):
            return np.std(x - x.mean())
        mean_deviation = typical_price.rolling(window=window).apply(custom_std, raw=False)
        cci = (typical_price - moving_average) / (0.015 * mean_deviation)
        return cci

    def calculate_bollinger_bands(self, df, window=20):
        """计算布林带（Bollinger Bands）"""
        moving_average = df['close'].rolling(window=window).mean()
        std_dev = df['close'].rolling(window=window).std()
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
        return moving_average, upper_band, lower_band

    def calculate_technical_indicators(self, symbol, update_type='all'):
        """计算技术指标"""
        try:
            df = self.symbol_hist_data.copy()
            if df is None:
                logging.warning(f"No historical data for {symbol}")
                return
            df.rename(columns={'日期': 'trade_date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高':'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'turnover', 
                    '振幅': 'amplitude',
                    '涨跌幅': 'price_change_percentage',
                    '涨跌额': 'price_change_amount',
                    '换手率': 'turnover_rate'}, inplace=True)
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
            
            # 新增技术指标计算
            df['rsi'] = self.calculate_rsi(df)
            df['obv'] = self.calculate_obv(df)
            df['vwap'] = self.calculate_vwap(df)
            df['cci'] = self.calculate_cci(df)
            df['bollinger_mavg'], df['bollinger_upper'], df['bollinger_lower'] = self.calculate_bollinger_bands(df)
            
            df['symbol'] = symbol
            cols = ['symbol', 'trade_date', 'open', 'close', 'high', 'low', 'volume', 
                'turnover', 'amplitude', 'price_change_percentage', 'price_change_amount', 'turnover_rate',
                'ma_5', 'ma_20', 'macd', 'kdj_k', 'rsi', 'obv', 'vwap', 
                'cci', 'bollinger_mavg', 'bollinger_upper', 'bollinger_lower']
            
            volatility = self.calculate_volatility(symbol)
            volatility_cols = ['hist_volatility_30d', 'hist_volatility_5d',
                      'beta_252d', 'atr_14d']
            if volatility is not None:
                df = pd.merge(df, volatility, on='trade_date', how='outer')
            else:
                df[volatility_cols] = np.nan
            cols += volatility_cols

            if update_type == 'all':
                pass 
            elif update_type == 'daily':
                start_date = pd.to_datetime(self.daily_start_date).date()
                end_date = pd.to_datetime('today').date()
                df = df[df['trade_date'].between(start_date, end_date)]
            df[cols].to_sql('stock_zh_a_hist', self.engine, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"Technical indicator error for {symbol}: {str(e)}")

    def initialize_hist_data(self):
        """初始化全量历史数据"""
        self.clear()
        
        stocks = self.get_all_stocks()
        
        for symbol in stocks:
            if 'unknown' in symbol:
                continue
            time.sleep(30)
            self.symbol_hist_data = self.get_hist_data(symbol)
            self._init_single_stock(symbol)
        
        # TODO: 宏观及融资融券数据特征，后续考虑拆分出来单独使用
        # self._init_macro_data()
        # self._init_margin_data()
        # self.update_sector_indices()  # 新增：初始化行业指数数据

    def _init_single_stock(self, symbol):
        """单只股票历史数据初始化"""
        logging.info(f"Processing {symbol}")
        try:
            self.update_valuation_factors(symbol)
            self.calculate_growth_factors(symbol)
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
        szse_df = ak.stock_margin_szse(date=start_date)
        
        margin_df = pd.concat([sse_df, szse_df])
        margin_df.to_sql('margin_data', self.engine, if_exists='append', index=False)
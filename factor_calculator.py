from sqlalchemy import text

class FullFactorCalculator:

    def initialize_hist_data(self):
        """初始化全量历史数据"""
        # 创建表
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS valuation_factors (
            symbol VARCHAR(10) NOT NULL, -- 证券代码
            trade_date DATE NOT NULL, -- 交易日期
            pe FLOAT, -- 市盈率
            pe_ttm FLOAT, -- 市盈率（TTM）
            pb FLOAT, -- 市净率
            dv_ratio FLOAT, -- 股息率
            dv_ttm FLOAT, -- 股息率（TTM）
            ps FLOAT, -- 市销率
            ps_ttm FLOAT, -- 市销率（TTM）
            total_mv FLOAT, -- 总市值
            PRIMARY KEY (symbol, trade_date),
            INDEX idx_symbol (symbol),
            INDEX idx_trade_date (trade_date)
        );

        CREATE TABLE IF NOT EXISTS growth_factors (
            symbol VARCHAR(10) NOT NULL, -- 证券代码
            announcement_date DATE NOT NULL, -- 公告日期
            net_profit_growth FLOAT, -- 净利润增长率
            revenue_growth FLOAT, -- 营业收入增长率
            rd_expense_growth FLOAT, -- 研发费用增长率
            PRIMARY KEY (symbol, announcement_date),
            INDEX idx_symbol (symbol),
            INDEX idx_announcement_date (announcement_date)
        );

        CREATE TABLE IF NOT EXISTS volatility_factors (
            symbol VARCHAR(10) NOT NULL, -- 证券代码
            trade_date DATE NOT NULL, -- 交易日期
            hist_volatility_30d FLOAT, -- 30日历史波动率
            hist_volatility_5d FLOAT, -- 5日历史波动率
            beta_252d FLOAT, -- 252日贝塔值
            atr_14d FLOAT, -- 14日平均真实波幅
            PRIMARY KEY (symbol, trade_date),
            INDEX idx_symbol (symbol),
            INDEX idx_trade_date (trade_date)
        );

        CREATE TABLE IF NOT EXISTS quality_factors (
            symbol VARCHAR(10) NOT NULL, -- 证券代码
            announcement_date DATE NOT NULL, -- 公告日期
            debt_to_asset FLOAT, -- 资产负债率
            cash_flow_ratio FLOAT, -- 现金流比率
            accruals_ratio FLOAT, -- 应计收入比率
            PRIMARY KEY (symbol, announcement_date),
            INDEX idx_symbol (symbol),
            INDEX idx_announcement_date (announcement_date)
        );

        CREATE TABLE IF NOT EXISTS technical_indicators (
            symbol VARCHAR(10) NOT NULL, -- 证券代码
            date DATE NOT NULL, -- 日期
            ma_5 FLOAT, -- 5日均线
            ma_20 FLOAT, -- 20日均线
            macd FLOAT, -- MACD
            kdj_k FLOAT, -- KDJ_K
            PRIMARY KEY (symbol, date),
            INDEX idx_symbol (symbol),
            INDEX idx_date (date)
        );

        CREATE TABLE IF NOT EXISTS macro_data (
            date DATE NOT NULL, -- 日期
            value FLOAT, -- 值
            indicator VARCHAR(10) NOT NULL, -- 指标
            PRIMARY KEY (date, indicator),
            INDEX idx_date (date),
            INDEX idx_indicator (indicator)
        );

        CREATE TABLE IF NOT EXISTS margin_data (
            date DATE NOT NULL, -- 日期
            market VARCHAR(10) NOT NULL, -- 市场
            margin_balance FLOAT, -- 融资余额
            margin_purchase FLOAT, -- 融资买入
            repayment_of_margin FLOAT, -- 融资偿还
            short_balance FLOAT, -- 融券余额
            short_sale FLOAT, -- 融券卖出
            repayment_of_short FLOAT, -- 融券偿还
            PRIMARY KEY (date, market),
            INDEX idx_date (date),
            INDEX idx_market (market)
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
        mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.std(x - moving_average.iloc[x.index.get_loc(x.name)]), raw=False)
        cci = (typical_price - moving_average) / (0.015 * mean_deviation)
        return cci

    def calculate_bollinger_bands(self, df, window=20):
        """计算布林带（Bollinger Bands）"""
        moving_average = df['close'].rolling(window=window).mean()
        std_dev = df['close'].rolling(window=window).std()
        upper_band = moving_average + (2 * std_dev)
        lower_band = moving_average - (2 * std_dev)
        return moving_average, upper_band, lower_band

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
            
            # 新增技术指标计算
            df['rsi'] = self.calculate_rsi(df)
            df['obv'] = self.calculate_obv(df)
            df['vwap'] = self.calculate_vwap(df)
            df['cci'] = self.calculate_cci(df)
            df['bollinger_mavg'], df['bollinger_upper'], df['bollinger_lower'] = self.calculate_bollinger_bands(df)
            
            df['symbol'] = symbol
            df[['symbol', 'date', 'ma_5', 'ma_20', 'macd', 'kdj_k', 'rsi', 'obv', 'vwap', 'cci', 'bollinger_mavg', 'bollinger_upper', 'bollinger_lower']].to_sql(
                'stock_zh_a_hist', self.engine, if_exists='append', index=False)
        except Exception as e:
            logging.error(f"Technical indicator error for {symbol}: {str(e)}")

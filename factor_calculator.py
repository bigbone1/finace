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
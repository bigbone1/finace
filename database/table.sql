### 1. `valuation_factors`
存储估值因子数据。
CREATE TABLE valuation_factors (
    symbol VARCHAR(10) NOT NULL, -- 股票代码
    trade_date DATE NOT NULL, -- 交易日期
    pe FLOAT, -- 市盈率（PE）
    pe_ttm FLOAT, -- 滚动市盈率（PE_TTM）
    pb FLOAT, -- 市净率（PB）
    dv_ratio FLOAT, -- 股息率
    dv_ttm FLOAT, -- 滚动股息率
    ps FLOAT, -- 市销率（PS）
    ps_ttm FLOAT, -- 滚动市销率
    total_mv FLOAT, -- 总市值
    PRIMARY KEY (symbol, trade_date),
    INDEX idx_symbol (symbol),
    INDEX idx_trade_date (trade_date)
);

### 2. `growth_factors`
存储成长因子数据。
CREATE TABLE growth_factors (
    symbol VARCHAR(10) NOT NULL, -- 股票代码
    announcement_date DATE NOT NULL, -- 公告日期
    net_profit_growth FLOAT, -- 净利润增长率
    revenue_growth FLOAT, -- 营业收入增长率
    rd_expense_growth FLOAT, -- 研发费用增长率
    PRIMARY KEY (symbol, announcement_date),
    INDEX idx_symbol (symbol),
    INDEX idx_announcement_date (announcement_date)
);

### 3. `stock_zh_a_hist`    存储波动率因子数据。存储技术指标数据。
CREATE TABLE stock_zh_a_hist (
    symbol VARCHAR(10) NOT NULL, -- 股票代码
    trade_date DATE NOT NULL, -- 交易日期
    open FLOAT, -- 开盘价
    close FLOAT, -- 收盘价
    high FLOAT, -- 最高价
    low FLOAT, -- 最低价
    volume FLOAT, -- 成交量
    turnover FLOAT, -- 成交额
    amplitude FLOAT, -- 振幅
    price_change_percentage FLOAT, -- 涨跌幅
    price_change_amount FLOAT, -- 涨跌额
    turnover_rate FLOAT, -- 换手率
    hist_volatility_30d FLOAT, -- 30日历史波动率
    hist_volatility_5d FLOAT, -- 5日历史波动率
    beta_252d FLOAT, -- 252日贝塔值
    atr_14d FLOAT, -- 14日平均真实波幅（ATR）
    ma_5 FLOAT, -- 5日移动平均线
    ma_20 FLOAT, -- 20日移动平均线
    macd FLOAT, -- MACD指标
    kdj_k FLOAT, -- KDJ指标中的K值
    rsi FLOAT,
    obv FLOAT,
    vwap FLOAT,
    cci FLOAT,
    bollinger_mavg FLOAT,
    bollinger_upper FLOAT,
    bollinger_lower FLOAT,
    PRIMARY KEY (symbol, trade_date),
    INDEX idx_symbol (symbol),
    INDEX idx_trade_date (trade_date)
);
 'cci', 'bollinger_mavg', 'bollinger_upper', 'bollinger_lower']
### 4. `quality_factors`
CREATE TABLE quality_factors (
    symbol VARCHAR(10) NOT NULL, -- 股票代码
    announcement_date DATE NOT NULL, -- 公告日期
    debt_to_asset FLOAT, -- 资产负债率
    cash_flow_ratio FLOAT, -- 现金流比率
    accruals_ratio FLOAT, -- 应计比率
    PRIMARY KEY (symbol, announcement_date),
    INDEX idx_symbol (symbol),
    INDEX idx_announcement_date (announcement_date)
);

### 5. `market_indices`
存储市场指数数据。

### 6. `macro_data`
存储宏观经济数据。
CREATE TABLE macro_data (
    date DATE NOT NULL, -- 日期
    value FLOAT, -- 数据值
    indicator VARCHAR(10) NOT NULL, -- 指标名称
    PRIMARY KEY (date, indicator),
    INDEX idx_date (date),
    INDEX idx_indicator (indicator)
);

### 7. `margin_data`
存储融资融券历史数据。
CREATE TABLE margin_data (
    date DATE NOT NULL, -- 日期
    market VARCHAR(10) NOT NULL, -- 市场（如沪市、深市）
    margin_balance FLOAT, -- 融资余额
    margin_purchase FLOAT, -- 融资买入额
    repayment_of_margin FLOAT, -- 融资偿还额
    short_balance FLOAT, -- 融券余额
    short_sale FLOAT, -- 融券卖出量
    repayment_of_short FLOAT, -- 融券偿还量
    PRIMARY KEY (date, market),
    INDEX idx_date (date),
    INDEX idx_market (market)
);

### 行业指数数据。
CREATE TABLE sector_indices (
    index_code VARCHAR(10) NOT NULL, -- 指数代码
    trade_date DATE NOT NULL, -- 交易日期
    open FLOAT, -- 开盘价
    close FLOAT, -- 收盘价
    high FLOAT, -- 最高价
    low FLOAT, -- 最低价
    volume FLOAT, -- 成交量
    turnover FLOAT, -- 成交额
    PRIMARY KEY (index_code, trade_date),
    INDEX idx_index_code (index_code),
    INDEX idx_trade_date (trade_date)
);
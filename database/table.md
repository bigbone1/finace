### 1. `valuation_factors`
存储估值因子数据。

```sql
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
```

### 2. `growth_factors`
存储成长因子数据。

```sql
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
```

### 3. `volatility_factors`
存储波动率因子数据。

```sql
CREATE TABLE volatility_factors (
    symbol VARCHAR(10) NOT NULL, -- 股票代码
    trade_date DATE NOT NULL, -- 交易日期
    hist_volatility_30d FLOAT, -- 30日历史波动率
    hist_volatility_5d FLOAT, -- 5日历史波动率
    beta_252d FLOAT, -- 252日贝塔值
    atr_14d FLOAT, -- 14日平均真实波幅（ATR）
    PRIMARY KEY (symbol, trade_date),
    INDEX idx_symbol (symbol),
    INDEX idx_trade_date (trade_date)
);
```

### 4. `quality_factors`
存储质量因子数据。

```sql
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
```

### 5. `technical_indicators`
存储技术指标数据。

```sql
CREATE TABLE technical_indicators (
    symbol VARCHAR(10) NOT NULL, -- 股票代码
    date DATE NOT NULL, -- 日期
    ma_5 FLOAT, -- 5日移动平均线
    ma_20 FLOAT, -- 20日移动平均线
    macd FLOAT, -- MACD指标
    kdj_k FLOAT, -- KDJ指标中的K值
    PRIMARY KEY (symbol, date),
    INDEX idx_symbol (symbol),
    INDEX idx_date (date)
);
```

### 6. `macro_data`
存储宏观经济数据。

```sql
CREATE TABLE macro_data (
    date DATE NOT NULL, -- 日期
    value FLOAT, -- 数据值
    indicator VARCHAR(10) NOT NULL, -- 指标名称
    PRIMARY KEY (date, indicator),
    INDEX idx_date (date),
    INDEX idx_indicator (indicator)
);
```

### 7. `margin_data`
存储融资融券历史数据。

```sql
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
```

### 解释
- **`PRIMARY KEY`**: 每个表都有一个主键，用于唯一标识每一条记录。
- **`INDEX`**: 为常用的查询字段添加索引，以提高查询性能。
  - `symbol` 和 `trade_date` 在多个表中被频繁查询，因此添加了索引。
  - `announcement_date` 在 `growth_factors` 和 `quality_factors` 表中被频繁查询，因此添加了索引。
  - `date` 在 `technical_indicators` 表中被频繁查询，因此添加了索引。
  - `indicator` 和 `market` 在 `macro_data` 和 `margin_data` 表中被频繁查询，因此添加了索引。

通过添加这些索引，可以显著提高对这些表的查询效率。
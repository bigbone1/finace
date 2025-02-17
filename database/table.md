为了提高查询性能，可以在表中添加适当的索引。以下是为每个表添加索引后的建表语句：

### 1. `valuation_factors`
存储估值因子数据。

```sql
CREATE TABLE valuation_factors (
    symbol VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    pe FLOAT,
    pb FLOAT,
    ps FLOAT,
    dividend_yield FLOAT,
    total_mv FLOAT,
    PRIMARY KEY (symbol, trade_date),
    INDEX idx_symbol (symbol),
    INDEX idx_trade_date (trade_date)
);
```

### 2. `growth_factors`
存储成长因子数据。

```sql
CREATE TABLE growth_factors (
    symbol VARCHAR(10) NOT NULL,
    announcement_date DATE NOT NULL,
    net_profit_growth FLOAT,
    revenue_growth FLOAT,
    rd_expense_growth FLOAT,
    PRIMARY KEY (symbol, announcement_date),
    INDEX idx_symbol (symbol),
    INDEX idx_announcement_date (announcement_date)
);
```

### 3. `volatility_factors`
存储波动率因子数据。

```sql
CREATE TABLE volatility_factors (
    symbol VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    hist_volatility_30d FLOAT,
    hist_volatility_5d FLOAT,
    beta_252d FLOAT,
    atr_14d FLOAT,
    PRIMARY KEY (symbol, trade_date),
    INDEX idx_symbol (symbol),
    INDEX idx_trade_date (trade_date)
);
```

### 4. `quality_factors`
存储质量因子数据。

```sql
CREATE TABLE quality_factors (
    symbol VARCHAR(10) NOT NULL,
    announcement_date DATE NOT NULL,
    debt_to_asset FLOAT,
    cash_flow_ratio FLOAT,
    accruals_ratio FLOAT,
    PRIMARY KEY (symbol, announcement_date),
    INDEX idx_symbol (symbol),
    INDEX idx_announcement_date (announcement_date)
);
```

### 5. `technical_indicators`
存储技术指标数据。

```sql
CREATE TABLE technical_indicators (
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    ma_5 FLOAT,
    ma_20 FLOAT,
    macd FLOAT,
    kdj_k FLOAT,
    PRIMARY KEY (symbol, date),
    INDEX idx_symbol (symbol),
    INDEX idx_date (date)
);
```

### 6. `macro_data`
存储宏观经济数据。

```sql
CREATE TABLE macro_data (
    date DATE NOT NULL,
    value FLOAT,
    indicator VARCHAR(10) NOT NULL,
    PRIMARY KEY (date, indicator),
    INDEX idx_date (date),
    INDEX idx_indicator (indicator)
);
```

### 7. `margin_data`
存储融资融券历史数据。

```sql
CREATE TABLE margin_data (
    date DATE NOT NULL,
    market VARCHAR(10) NOT NULL,
    margin_balance FLOAT,
    margin_purchase FLOAT,
    repayment_of_margin FLOAT,
    short_balance FLOAT,
    short_sale FLOAT,
    repayment_of_short FLOAT,
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
CREATE TABLE valuation_factors (
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
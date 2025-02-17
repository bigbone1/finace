import sys
sys.path.append(r'D:\python\finace')
from libs.mysql import MyEngine
import pandas as pd



column_mapping = {
    "股票代码": "stock_code",
    "日期": "date",
    "开盘": "open_price",
    "收盘": "close_price",
    "最高": "high_price",
    "最低": "low_price",
    "成交量": "volume",
    "成交额": "turnover",
    "振幅": "amplitude",
    "涨跌幅": "price_change_percentage",
    "涨跌额": "price_change_amount",
    "换手率": "turnover_rate"
}

# 创建包含'name_cn'和'name_en'两列的DataFrame
df = pd.DataFrame({
    'name_cn': list(column_mapping.keys()),
    'name_en': list(column_mapping.values())
})
MyEngine().to_mysql(df, 'name_map')
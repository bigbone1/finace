import pandas as pd
from .mysql import MyEngine


def code_convet(symbol, res_with_prefix=True):
    if not res_with_prefix:
        return int(symbol.split('.')[1])
    else:
        str_symbol = str(symbol).rjust(6, '0')
        return ['sz.'+str_symbol, 'sh.'+str_symbol]
    
def convert_names(df: pd.DataFrame):
    sql3 = 'select * FROM name_map'
    name_df = MyEngine().read_sql_query(sql3)
    valid_df = name_df[name_df.name_cn.isin(df.columns)]
    cols_dict = valid_df.set_index('name_cn').to_dict()['name_en']
    df.rename(columns=cols_dict, inplace=True)
    return df

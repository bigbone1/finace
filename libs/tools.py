import pandas as pd
from .mysql import MyEngine


def convert_names(df: pd.DataFrame):
    sql3 = 'select * FROM name_map'
    name_df = MyEngine().read_sql_query(sql3)
    valid_df = name_df[name_df.name_cn.isin(df.columns)]
    cols_dict = valid_df.set_index('name_cn').to_dict()['name_en']
    df.rename(columns=cols_dict, inplace=True)
    return df

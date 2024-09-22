import pandas as pd
from sqlalchemy import create_engine



class MyEngine():
    def __init__(self, database: str = 'finance') -> None:
        username = 'root'
        password = ''
        host = '127.0.0.1:3306'
        database = database
        db_url = f'mysql+pymysql://{username}:{password}@{host}/{database}'
        self.engine = create_engine(db_url)

    def to_mysql(self, data: pd.DataFrame, table_name: str, if_exists='append', index=False, index_label=None):
        # 替换以下参数为你的数据库信息
        # 替换'your_table_name'为你的表名
        data.to_sql(table_name, con=self.engine, if_exists=if_exists, index=index, index_label=index_label)

        print('=======================save success===============================')

    def read_sql_query(self, sql):
        return pd.read_sql_query(sql, con=self.engine)
    
    def create_sql(self, columns, table_name, start_date, end_date, codes=[]):
        sql_cols = ', '.join(columns)
        sql = f'select {sql_cols} from {table_name} \
                    where date between "{start_date}" and "{end_date}"'
        if len(codes) > 0:
            sql_codes = ', '.join(codes)
            sql = sql + f' and stock_code in ({sql_codes})'
        return sql
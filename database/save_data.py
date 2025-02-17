import sys
sys.path.append(r'D:\python\finace')
from libs.mysql import to_mysql
import pandas as pd

data = pd.read_csv('industry_cons_em_details_2.csv')
to_mysql(data, 'industry_cons_em_details')
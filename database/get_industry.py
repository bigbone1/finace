import pandas as pd
from time import sleep
import akshare as ak

error_industry = pd.read_csv('error_industry.csv')
industry_cons_em_details = pd.read_csv('industry_cons_em_details.csv')

error = []
for i in enumerate(error_industry['error_industry'].values):
    print("processing: ", i[1], "{}/{}".format(i[0]+1, error_industry.shape[0]))
    
    try:
        industry_cons_em = ak.stock_board_industry_cons_em(i[1])
        industry_cons_em['板块名称'] = i[1]
        if industry_cons_em_details.empty:
            industry_cons_em_details = industry_cons_em
        else:
            industry_cons_em_details = pd.concat([industry_cons_em_details, industry_cons_em], ignore_index=True)
        print("success!")
    except Exception as e:
        error.append(i[1])
        print('failed!')
        print(str(e))
        
    sleep(10)


pd.DataFrame(data={'error_industry': error}).to_csv('error_industry_2.csv')
industry_cons_em_details.to_csv('industry_cons_em_details_2.csv')
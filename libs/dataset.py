import datetime
import pathlib
import pickle
from sklearn.discriminant_analysis import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from libs import MyEngine
from libs.tools import code_convet

# 假设 train_dataset 是一个包含所有数据和标签的列表或数组
# 例如: train_dataset = [(data1, label1), (data2, label2), ...]
# 创建新的数据集
class BalancedDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = np.array(labels).reshape(-1, 1)
        self.labels = torch.from_numpy(self.labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def balanced_dataset(train_dataset):
    # 分离数据和标签
    data = [x[0] for x in train_dataset]
    labels = [x[1] for x in train_dataset]

    # 筛选出标签为0和1的数据
    data_0 = [d for d, l in zip(data, labels) if l == 0]
    data_1 = [d for d, l in zip(data, labels) if l == 1]

    # 确保两个类别的数据数量相等
    min_size = min(len(data_0), len(data_1))
    data_0 = data_0[:min_size]
    data_1 = data_1[:min_size]

    # 合并数据集
    balanced_data = data_0 + data_1
    balanced_labels = [0] * min_size + [1] * min_size
    return BalancedDataset(balanced_data, balanced_labels)

class HighFreqDataset(Dataset):
    def __init__(self, start_date, end_date,
                 input_days=5, max_return_days=5, max_return=20,
                 mode='save',
                 multi=False):
        self.start_date = start_date
        self.end_date = end_date
        self.input_days = input_days   
        self.max_return_days = max_return_days   # 计算回报的天数 尾盘最后半小时买入
        self.max_return_thresh = max_return      # 回报最小阈值20%
        self.multi = multi

    def save(self, save_dir='./data/high_freq_dataset/'):
        sql = 'select * from high_freq_labels where date between "{}" and "{}"'.format(self.start_date, self.end_date)
        labels = MyEngine().read_sql_query(sql)
        if not self.multi:
            for code in labels['stock_code'].unique():
                self._save_core(code, labels, save_dir)

    def read(self, read_dir='./data/high_freq_dataset/'):
        feats = []
        labels = []
        for file in pathlib.Path(read_dir).rglob('*.pkl'):
            with open('data.pkl', 'rb') as f:
                feat, label = pickle.load(f)
                feats.append(feat)
                labels.append(label)

        self.data = torch.from_numpy(np.concatenate(feats, axis=0)).float()
        self.labels = torch.from_numpy(np.concatenate(labels, axis=0)).float()

    def _init_dataset(self):
        sql = 'select * from high_freq_labels where date between "{}" and "{}"'.format(self.start_date, self.end_date)
        labels = MyEngine().read_sql_query(sql)

        if self.mode == 'save':
            if not self.multi:
                for code in labels['stock_code'].unique():
                    self._save_core(code, labels)
                    break

    def _save_core(self, code, labels, save_dir):
        codes = code_convet(code, res_with_prefix=True)
        sql = 'select * from stock_zh_5_minute where date between "{}" and "{}" and code = "{}"'\
            .format(self.start_date, self.end_date, codes[0])
        features = MyEngine().read_sql_query(sql)
        if features.empty:
            sql = 'select * from stock_zh_5_minute where date between "{}" and "{}" and code = "{}"'\
                .format(self.start_date, self.end_date, codes[1])
            features = MyEngine().read_sql_query(sql)

        if features.empty:
            return
        features.drop(columns=['adjustflag'], inplace=True)
        features['stock_code'] = features['code'].apply(lambda x: x.split('.')[1]).astype(int)

        feat, label = self._dataset_core(features, labels)
        true_label_num = np.np.count_nonzero(label == 1)

        self._to_pkl(data=(feat, label), name=save_dir+f'{code}_{true_label_num}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def _dataset_core(self, features, labels):
        features['date'] = features['date'].astype(str)
        data = pd.merge(features, labels, on=['stock_code', 'date'])
        data.sort_values(by=['code', 'date', 'time'], inplace=True)
        data['time'] = data['time'].apply(lambda x: datetime.strptime(x, "%Y%m%d%H%M%S%f"))
        feat_list = []
        label_list = []
        feat_cols = ['open', 'close', 'high', 'low', 'volume', 'amount']
        for code, df in data.groupby('stock_code'):
            for row in df.iterrows():
                df_temp = df[df['date'].between(row['start_date'], row['end_date'])]
                cutoff_time = pd.Timestamp(f"{row['end_date']} 14:30:00")
                df_temp = df_temp[df_temp['time'] < cutoff_time]
                if df_temp.shape[0] < 20 * self.input_days:
                    continue 
                df_temp.set_index('time', inplace=True)
                df_temp[feat_cols] = df_temp[feat_cols].interpolate(method='linear')
                temp  = self._scaled(df_temp[feat_cols].values)
                feat_list.appned(temp)
                label_list.append(row['label'])

        feat_res = np.concatenate(feat_list, axis=0)
        label_res = np.array(label_list)
        return feat_res, label_res.reshape(-1, 1)
    
    def _scaled(self, data):
        # 进行z score归一化
        scaler1 = StandardScaler()
        data1_scaled = scaler1.fit_transform(data.reshape(1, data.shape[0], data.shape[1]))
        return data1_scaled
    
    def _to_pkl(self, data, name):
        with open(f'{name}.pkl', 'wb') as f:
            pickle.dump(data, f)
    
if __name__ == '__main__':
    
    dataset = HighFreqDataset('2024-06-01', '2024-12-31')
    dataset.save()
    dataset.read()
    test_data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    for batch in test_data_loader:
        print(batch[0].shape)  # 假设 batch[0] 包含了数据，batch[1] 包含了标签（如果有的话）
        break  # 只查看第一个批次的形状
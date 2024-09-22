import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys

from libs.dataset import balanced_dataset
sys.path.append(r'D:\python\finace')
from libs import MyEngine

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(32 * 2, 128)  # 假设卷积层输出尺寸为 (batch_size, 32, 80)
        self.fc2 = nn.Linear(128, 1)  # 二分类问题

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # 展平卷积输出
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return x
        return torch.sigmoid(x)
    
class CNN_RNN(nn.Module):
    def __init__(self):
        super(CNN_RNN, self).__init__()
        self.conv1 = nn.Conv1d(16, 32, kernel_size=3)
        self.conv1_drop = nn.Dropout1d()  # 增加Dropout比例
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3)
        self.batch_norm = nn.BatchNorm1d(32)  # 添加批量归一化层
        self.conv2_drop = nn.Dropout1d()  # 为第二个卷积层添加Dropout
        self.rnn = nn.GRU(32, 64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)

        self._initialize_weights()
        

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.conv1_drop(x)
        x = F.max_pool1d(x, 2)
        
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.conv2_drop(x)
        x = F.max_pool1d(x, 2)
        
        # 确保这里的形状与max pooling的结果一致
        # 假设序列长度在经过两次池化后为 8
        x = x.view(-1, 8, 32)  
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        # x = torch.sigmoid(x)
        return x


# 构建PyTorch数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, data1, earnings, window=40, stride=5, device='cpu'):
        self.window = window
        self.stride = stride
        data, labels = self.__init_data__(data1, earnings)
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
    
    def __init_data__(self, data1, earnings):
        # 构建数据集
        data1_res = []
        labels_res = []
        for i in range(0, len(data1) - self.window, self.stride):
            data1_res.append(data1[i:i+self.window])
            labels_res.append(earnings[i+self.window-1])
        data1_res = np.array(data1_res)
        labels_res = np.array(labels_res)

        # 进行z score归一化
        scaler1 = StandardScaler()
        data1_scaled = scaler1.fit_transform(data1_res.reshape(-1, data1.shape[1])).reshape(-1, self.window, data1.shape[1])
        return np.swapaxes(data1_scaled, 1, 2), labels_res.reshape(-1, 1)
    

def train(data, earnings, max_echos=50):
    """
    Trains a CNN-RNN model on a time series dataset.

    Parameters:
        data1 (pandas.DataFrame): The first dataset.
        data2 (pandas.DataFrame): The second dataset.
        earnings (pandas.DataFrame): The earnings dataset.

    Returns:
        None
    """
    # 构建数据集
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = TimeSeriesDataset(data.values,  earnings.values, window=5, stride=1, device=device)
    dataset = balanced_dataset(dataset)
    
    # Split the dataset into training and testing set
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
  
    # Create data loaders for training and testing
    batch_size = 128
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # for batch in test_data_loader:
    #     print(batch[0].shape)  # 假设 batch[0] 包含了数据，batch[1] 包含了标签（如果有的话）
    #     break  # 只查看第一个批次的形状
    # exit()
    # Initialize the model, optimizer, and loss function
    model = SimpleCNN()
    # 正确的初始化方法示例
    # nn.init.kaiming_uniform_(model.parameters())
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = FocalLoss()

    # Train the model
    train_losses = []
    test_losses = []
    best_loss = float('inf')
    best_epoch = 0
    for epoch in range(max_echos):
        # Train the model in each epoch
        model.train()
        model = model.to(device)
        for batch in train_data_loader:
            model = model.to(device)
            data, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        train_loss = loss.item()
        train_losses.append(train_loss)
        
        # Evaluate the model in each epoch
        model.eval()
        test_loss = 0
        model = model.to('cpu')
        with torch.no_grad():
            for batch in test_data_loader:
                data, labels = batch
                outputs = model(data)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        test_loss /= len(test_data_loader)
        test_losses.append(test_loss)
        
        # Save the best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Test Loss: {test_loss}')

    # Plot the training and testing loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    # plt.show()
    plt.savefig('loss_curve.png')

    print(f'Best Epoch: {best_epoch+1}, Best Test Loss: {best_loss}')

def prepare_data(start_date, end_date, target_code_num, train=True):
    sql3 = f'select stock_code, date, open_price, close_price \
        from stock_zh_a_hist_daily where date between "{start_date}" and "{end_date}"'
    predicts = MyEngine().read_sql_query(sql3)
    predicts.sort_values(by=['date'], inplace=True)
    target_code = predicts['stock_code'].unique()[:target_code_num]
    def func_(x):
        return x.iloc[-1]
    predicts = predicts[predicts['stock_code'].isin(target_code)]
    predicts['close_price_next_5'] = predicts.groupby(['stock_code'])['close_price'].transform(lambda x: x.rolling(5).apply(func_))
    predicts['cum_sum_5'] = (predicts['close_price_next_5'] - predicts['open_price']) / predicts['open_price'] * 100

    sql4 = f'select stock_code, date, open_price, close_price, volume, \
        turnover, amplitude, price_change_percentage, price_change_amount, \
            turnover_rate  from stock_zh_a_hist_daily where date between "{start_date}" and "{end_date}" and stock_code in '
    for i, code in enumerate(target_code):
        if i == (len(target_code)-1):
            sql4 = sql4 + str(code) + ')'
        elif i == 0:
            sql4 = sql4 + '(' + str(code) + ', '
        else:
            sql4 = sql4 + str(code) + ', '
    inputs1 = MyEngine().read_sql_query(sql4)

    sql5 = f'select stock_code, date, market_capitalization, earnings_ratio_ttm, \
        earnings_ratio_static, book_ratio, cash_flow_ratio from stock_zh_valuation_baidu \
            where date between "{start_date}" and "{end_date}" and stock_code in '
    for i, code in enumerate(target_code):
        if i == (len(target_code)-1):
            sql5 = sql5 + str(code) + ')'
        elif i == 0:
            sql5 = sql5 + '(' + str(code) + ', '
        else:
            sql5 = sql5 + str(code) + ', '
    inputs2 = MyEngine().read_sql_query(sql5)

    sql6 = 'select 板块名称 as industry_name, 代码 as stock_code from industry_cons_em_details'
    inputs3 = MyEngine().read_sql_query(sql6)

    inputs1.sort_values(by=['stock_code', 'date'], inplace=True)
    inputs1.loc[:, 'date_seconds'] = pd.to_datetime(inputs1['date']).astype('int64')//1e9
    inputs1['price_change_percentage_next_day'] = inputs1['price_change_percentage'].apply(lambda x: 1 if x>5 else 0)
    # inputs1 = inputs1.groupby('stock_code').apply(lambda x: x['open_price']/x['open_price'].iloc[0])
    inputs1.dropna(inplace=True)
    # data1 = inputs1.drop(columns=['date'])
    # earnings = data1['price_change_percentage_next_day']
    # data3 = data1.drop(columns=['price_change_percentage_next_day'])

    data2 = pd.merge(inputs2, inputs3, on='stock_code', how='left')
    data2.sort_values(by=['stock_code', 'date'], inplace=True)
    data2 = data2.groupby('stock_code').apply(lambda x: x.set_index('date').asfreq('D').reset_index().fillna(method='ffill'))
    industry_map = inputs3['industry_name'].drop_duplicates().reset_index().drop('index', axis=1).reset_index().set_index('industry_name').to_dict()['index']
    data2.loc[:, 'industry_name'] = data2['industry_name'].apply(lambda x: industry_map[x])
    data2.loc[:, 'date_seconds'] = pd.to_datetime(data2['date']).astype('int64')//1e9
    data2.drop(columns=['date'], inplace=True)
    data2.reset_index(inplace=True, drop=True)
    temp = pd.merge(inputs1, data2, on=['stock_code', 'date_seconds'], how='left')
    data2 = temp.fillna(method='ffill').fillna(method='bfill')
    data2 = data2[data2['date_seconds'].isin(inputs1.date_seconds)]

    data2.sort_values(by=['stock_code', 'date_seconds'], inplace=True)
    earnings = data2['price_change_percentage_next_day']
    if train:
        data2.drop(columns=['price_change_percentage_next_day', 'date'], inplace=True)
    print(data2.shape, 'data shape')
    return data2, earnings

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=3, reduction='sum'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # 预测概率的补码
        F_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


if __name__ == '__main__':
    data, labels = prepare_data('2023-07-01', '2024-05-31', target_code_num=1000, train=True)
    train(data, labels, max_echos=10000)

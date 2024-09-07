import torch
from torch.utils.data import DataLoader

def predict(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            data, _ = batch
            outputs = model(data)
            predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    return predictions

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model

if __name__ == '__main__':
    from models.model import TimeSeriesDataset, prepare_data, CNN_RNN
    import matplotlib.pyplot as plt
    import pandas as pd

    data, labels = prepare_data('2023-07-01', '2024-05-31', target_code_num=100, train=False)
    cols = data.columns.to_list()
    cols.remove('date')
    cols.remove('price_change_percentage_next_day')
    dataset = TimeSeriesDataset(data[cols].values, labels.values, window=40, stride=1)
    # 构建数据加载器
    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # 加载模型的函数
    model = CNN_RNN()
    model = load_model(model, r'best_model.pth')
    data['predict'] = pd.Series(predict(model, data_loader))
    def func_(x):
        return (x['predict']==x['price_change_percentage_next_day']).sum() / x.shape[0] * 100
    res = data.groupby(['stock_code']).apply(lambda x: func_(x))
    res = pd.DataFrame(res).reset_index().rename(columns={0: 'accuracy'})
    res.sort_values(by=['accuracy'], inplace=True, ascending=False)
    name = 'accuracy'
    plt.clf()
    ax = res.iloc[:20, :].plot(x='stock_code', y='accuracy', kind='bar')
    plt.savefig(f'./figs/{name}.png')
    plt.close()

    for code, df in data.groupby('stock_code'):
        if code in res.iloc[:20, :]['stock_code'].values:
            plt.clf()
            ax = df.plot(x='date', y='price_change_percentage_next_day', kind='line')
            df.plot(x='date', y='predict', kind='line', ax=ax)
            plt.savefig(f'./figs/{code}.png')
            plt.close()

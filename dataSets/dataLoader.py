import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
from utils.timefeatures import time_features
# from configClass import config

# class ETTh1Dataset(Dataset):
#     def __init__(self, data, input_window, output_window, scaler=None):
#         self.data = data
#         self.input_window = input_window
#         self.output_window = output_window
#         self.scaler = scaler
#
#         if self.scaler:
#             self.data = self.scaler.fit_transform(self.data)
#
#     def __len__(self):
#         return len(self.data) - self.input_window - self.output_window + 1
#
#     def __getitem__(self, idx):
#         x = self.data[idx:idx + self.input_window]
#         y = self.data[idx + self.input_window:idx + self.input_window + self.output_window]
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)



# def load_data(filepath, CONFIG):
#     input_window = CONFIG.input_length
#     output_window = CONFIG.output_length
#     batch_size = CONFIG.batch_size
#     train_ratio = CONFIG.train_ratio
#
#     # Load the CSV data into a Pandas DataFrame
#     df = pd.read_csv(filepath, index_col='date', parse_dates=True)
#     df = df.fillna(method='ffill')  # Handle missing values by forward filling
#     target_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
#     # Extract the features you want to use
#     data = df[target_columns].values  # Assuming 'OT' is the target column
#
#     # Split the data into training and test sets
#     train_size = int(len(data) * train_ratio)
#     # train_data = data[:train_size]
#     train_data = data[len(data) - train_size:]
#
#     test_data = data[:len(data) - train_size]
#
#     # test_data = data[train_size:]
#     # train_data = data[train_size:]
#
#     # Standardize the data
#     scaler = StandardScaler()
#
#
#     # Create dataset and DataLoader for training and test sets
#     train_dataset = ETTh1Dataset(train_data, input_window=input_window, output_window=output_window, scaler=scaler)
#     test_dataset = ETTh1Dataset(test_data, input_window=input_window, output_window=output_window, scaler=scaler)
#     # print(train_dataset)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
#     # print(len(test_loader))
#     # print(len(train_loader))
#     return train_loader, test_loader


# dataset of ETTh1
class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # M or MS: drop date
        # S: only reserve target column
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

# seq_len == input_window
# pred_len == output_window
# what is label_len?
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

# what is embed and timeenc? embed is 'timeF' then timeenc is 1, date(include hour, day, month...)  is scaled to [-0.5, 0.5], else timeenc is 1 date is the true value of hour, day, weekday, month
# when features == 'S', only reserve target column and date column, else use all column
def data_provider(embed, batch_size, freq, root_path, data_path, seq_len, label_len, pred_len, features, target, num_workers, flag):
    Data = Dataset_ETT_hour
    timeenc = 0 if embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False  # fix bug
        batch_size = batch_size
        freq = freq
    elif flag == 'pred':
        # shuffle_flag = False
        # drop_last = False
        # batch_size = 1
        # freq = freq
        # Data = Dataset_Pred
        pass
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = batch_size
        freq = freq

    data_set = Data(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[seq_len, label_len, pred_len],
        features=features,
        target=target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    return data_set, data_loader

if __name__ == "__main__":
    # Specify the path to your ETTh1 data CSV file
    filepath = "ETTh1.csv"

    # Load the data
    input_window = 96  # Number of time steps for the input (for long-term forecasting)
    output_window = 32  # Number of time steps for the output (for long-term forecasting)
    batch_size = 16

    train_loader, test_loader = load_data(filepath, input_window, output_window, batch_size)

    # Iterate through the training DataLoader to see how batches are prepared
    for batch_idx, (x, y) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1} - X shape: {x.shape}, Y shape: {y.shape}")
        break  # Print only the first batch for demonstration purposes
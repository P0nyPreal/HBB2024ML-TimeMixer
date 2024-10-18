import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class ETTh1Dataset(Dataset):
    def __init__(self, data, input_window, output_window, scaler=None):
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.scaler = scaler

        if self.scaler:
            self.data = self.scaler.fit_transform(self.data)

    def __len__(self):
        return len(self.data) - self.input_window - self.output_window + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.input_window]
        y = self.data[idx + self.input_window:idx + self.input_window + self.output_window]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)



def load_data(filepath, input_window=1, output_window=1, batch_size=32, train_ratio=0.8):
    # Load the CSV data into a Pandas DataFrame
    df = pd.read_csv(filepath, index_col='date', parse_dates=True)
    df = df.fillna(method='ffill')  # Handle missing values by forward filling
    target_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    # Extract the features you want to use
    data = df[target_columns].values  # Assuming 'OT' is the target column

    # Split the data into training and test sets
    train_size = int(len(data) * train_ratio)
    train_data = data[:10000]
    test_data = data[10000:12500]

    # test_data = data[:train_size]
    # train_data = data[train_size:]

    # Standardize the data
    scaler = StandardScaler()


    # Create dataset and DataLoader for training and test sets
    train_dataset = ETTh1Dataset(train_data, input_window=input_window, output_window=output_window, scaler=scaler)
    test_dataset = ETTh1Dataset(test_data, input_window=input_window, output_window=output_window, scaler=scaler)
    # print(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    # print(len(test_loader))
    # print(len(train_loader))
    return train_loader, test_loader


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
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import yfinance as yf
import datetime
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import yfinance as yf
import datetime

class ExcessReturnDataset(Dataset):
    def __init__(self, seq_len=50, mode='train'):
        self.seq_len = seq_len

        # Date ranges
        start_date = '2019-01-01'
        end_date = '2024-12-18'

        btc_file = 'btc_data.csv'
        spy_file = 'spy_data.csv'

        # Check if local files exist
        if os.path.exists(btc_file) and os.path.exists(spy_file):
            btc_data = pd.read_csv(btc_file, index_col=0, parse_dates=True)
            spy_data = pd.read_csv(spy_file, index_col=0, parse_dates=True)
        else:
            # Download data from yfinance if files don't exist
            btc_data = yf.download('BTC-USD', start=start_date, end=end_date)
            spy_data = yf.download('SPY', start=start_date, end=end_date)

            # Save to CSV for future runs
            btc_data.to_csv(btc_file)
            spy_data.to_csv(spy_file)

        # If the files exist but are outdated or user wants to ensure no re-download:
        # You could load directly from CSV if you'd prefer:
        # btc_data = pd.read_csv(btc_file, index_col=0, parse_dates=True)
        # spy_data = pd.read_csv(spy_file, index_col=0, parse_dates=True)
        #
        # But here we assume using the downloaded dataframes is fine.

        # Align by joining on common dates
        combined = btc_data[['Close']].rename(columns={'Close': 'BTC_Close'}).join(
            spy_data[['Close']].rename(columns={'Close': 'SPY_Close'}), how='inner'
        )

        # Drop rows with any NaN values
        combined = combined.dropna()

        # Extract arrays
        btc_prices = combined['BTC_Close'].values
        spy_prices = combined['SPY_Close'].values
        dates = combined.index

        # Compute daily returns: return[t] = (price[t] / price[t-1]) - 1
        btc_returns = btc_prices[1:] / btc_prices[:-1] - 1
        spy_returns = spy_prices[1:] / spy_prices[:-1] - 1
        excess_returns = btc_returns - spy_returns
        returns_dates = dates[1:]

        # Define date boundaries for splits
        train_end = datetime.datetime(2021, 1, 1)
        val_end = datetime.datetime(2022, 1, 1)
        test_end = datetime.datetime(2024, 12, 18)

        if mode == 'train':
            mask = (returns_dates >= datetime.datetime(2019, 1, 1)) & (returns_dates < train_end)
        elif mode == 'val':
            mask = (returns_dates >= train_end) & (returns_dates < val_end)
        else:  # 'test'
            mask = (returns_dates >= val_end) & (returns_dates <= test_end)

        filtered_returns = excess_returns[mask]
        self.data = torch.tensor(filtered_returns, dtype=torch.float32)

        # Store corresponding dates for plotting or reference
        self.dates = returns_dates[mask]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]  # (seq_len,)
        y = self.data[idx + self.seq_len]  # scalar (next day's excess return)
        return x.unsqueeze(-1), y.unsqueeze(-1)  # (seq_len, 1), (1,)


#####################################
# LSTM Model
#####################################
class TwoLayerLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=1):
        super(TwoLayerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        preds = self.fc(last_out)
        return preds
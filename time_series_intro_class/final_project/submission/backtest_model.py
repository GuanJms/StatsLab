import torch
import matplotlib.pyplot as plt
from utils import ExcessReturnDataset, TwoLayerLSTM
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


# Check if GPU is available and set device accordingly
device = torch.device("cpu")
criterion_mse = nn.MSELoss()


#####################################
# Prepare Data
#####################################
batch_size = 32

train_dataset = ExcessReturnDataset(seq_len=50, mode='train')
val_dataset = ExcessReturnDataset(seq_len=50, mode='val')
test_dataset = ExcessReturnDataset(seq_len=50, mode='test')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

import os
import torch
import matplotlib.pyplot as plt

num_layers = 3
hidden_size = 64
optimal_epoch = 1400

# Assuming model, train_loader, and val_loader are already defined
saved_models_dir = "/Users/jamesguan/Project/StatsLab/time_series_intro_class/final_project/temp/"  # Replace with the actual path

# Define the matching prefix
optimal_weight_file = f"{num_layers}_layer_{hidden_size}_size_model_epoch_{optimal_epoch}.pth"


model = TwoLayerLSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)

model_path = os.path.join(saved_models_dir, optimal_weight_file)
state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # Correctly apply map_location here
model.load_state_dict(state_dict)  # Load the state_dict into the model

model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for i in range(len(test_dataset)):
        features, target = test_dataset[i]
        features = features.unsqueeze(0).to(device)  # Move to device
        pred = model(features)

        predictions.append(pred.item())
        actuals.append(target.item())


# Assume predictions and actuals are 1D lists of returns.
# For example, predictions[i] = 0.01 means a predicted 1% gain at time i.
# actuals[i] = 0.005 means an actual 0.5% gain at time i.
# Both predictions and actuals are returns, NOT prices.

import numpy as np

predictions = np.array(predictions)  # Predicted returns
actuals = np.array(actuals)  # Actual returns
dates = test_dataset.dates[50:]
print(len(dates))

prediction_sign = np.sign(predictions)  # +1 for long, -1 for short
daily_return = prediction_sign * actuals  # Element-wise multiplication

initial_capital = 1000.0
# Compute cumulative portfolio value over time
portfolio_value = np.zeros(len(daily_return))
daily_pnl = np.zeros(len(daily_return))
portfolio_value[0] = initial_capital
daily_pnl[0] = 0.0
for t in range(1, len(portfolio_value)):
    portfolio_value[t] = portfolio_value[t-1] * (1 + daily_return[t-1])
    daily_pnl[t] = daily_return[t-1] * portfolio_value[t-1]

# Accumulated PnL = portfolio_value - initial_capital
accumulated_pnl = portfolio_value - initial_capital

# Compute Sharpe ratio over time
sharpe_ratios = []
for t in range(1, len(daily_pnl) + 1):
    sub_period = daily_pnl[:t]
    mean_pnl = np.mean(sub_period)
    std_pnl = np.std(sub_period)
    sr = mean_pnl / std_pnl if std_pnl > 1e-12 else 0.0
    sharpe_ratios.append(sr)

# Compute maximum drawdown
running_max = np.maximum.accumulate(portfolio_value)
drawdown = running_max - portfolio_value
max_drawdown = np.max(drawdown)

print("Max Drawdown:", max_drawdown)

# Plotting
plt.figure(figsize=(15, 8))

plt.subplot(3, 1, 1)
plt.plot(dates, accumulated_pnl, label='Accumulated PnL', color='blue')
plt.title(f'Accumulated PnL Over Time (Final PnL: {accumulated_pnl[-1]:.4f}) - Starting Capital: {initial_capital}')
plt.xlabel('Time')
plt.ylabel('PnL')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(dates[30:], sharpe_ratios[30:], label='Sharpe Ratio', color='green')
plt.title(f'Sharpe Ratio Over Time (Final SR: {sharpe_ratios[-1]:.4f})- Starting Capital: {initial_capital}')
plt.xlabel('Time')
plt.ylabel('Sharpe Ratio')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(drawdown, label='Drawdown', color='red')
plt.title(f'Drawdown Over Time (Max DD: {max_drawdown:.4f}) - Starting Capital: {initial_capital}')
plt.xlabel('Time')
plt.ylabel('Drawdown')
plt.grid(True)
plt.legend()

plt.tight_layout()

# Save the figure before showing it
backtest_plot_filename = f'backtest_result/backtest_plot_{num_layers}_layers_{hidden_size}_hidden_size.png'
plt.savefig(backtest_plot_filename, dpi=300)  # Save with high resolution
plt.show()

# Jump the backtest stats to a dictionary for easy access for later creating table
backtest_stats = {
    'Max Drawdown': max_drawdown,
    'Final PnL': accumulated_pnl[-1].item(),
    'Final Sharpe Ratio': sharpe_ratios[-1].item(),
    'Num Layers': num_layers,
    'Hidden Size': hidden_size,
    'Optimal Epoch': optimal_epoch
}

# Write the backtest stats to a JSON file
import json

backtest_stats = 'backtest_result/backtest_stats.json'

# read the json file and append the new data to it, if it exists then overwrite it

import json
import os

# Backtest statistics
backtest_stats = {
    'Max Drawdown': max_drawdown,
    'Final PnL': accumulated_pnl[-1].item(),
    'Final Sharpe Ratio': sharpe_ratios[-1].item(),
    'Num Layers': num_layers,
    'Hidden Size': hidden_size,
    'Optimal Epoch': optimal_epoch
}

# File path
json_file_path = 'backtest_result/backtest_stats.json'

# Ensure the directory exists
os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

# Check if the JSON file already exists
if os.path.exists(json_file_path):
    # Read the existing data
    with open(json_file_path, 'r') as f:
        try:
            existing_data = json.load(f)
        except json.JSONDecodeError:
            # If the file is empty or corrupted, start with an empty list
            existing_data = []
else:
    # If the file doesn't exist, start with an empty list
    existing_data = []

# Update the JSON data (append the new stats)
# If `existing_data` is a list, append; otherwise, overwrite with a list
if isinstance(existing_data, list):
    existing_data.append(backtest_stats)
else:
    existing_data = [backtest_stats]

# Write the updated data back to the file
with open(json_file_path, 'w') as f:
    json.dump(existing_data, f, indent=4)

print(f"Backtest stats saved to {json_file_path}")

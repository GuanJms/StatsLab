from utils import ExcessReturnDataset, TwoLayerLSTM
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


# Check if GPU is available and set device accordingly
device = torch.device("cpu")
criterion_mse = nn.MSELoss()

def evaluate_metrics(model, data_loader):
    model.eval()
    mse_list = []
    dir_accuracy_list = []
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            preds = model(features)  # preds are returns directly
            mse = criterion_mse(preds, targets)

            predicted_returns = preds.squeeze()  # shape: (batch,)
            actual_returns = targets.squeeze()  # shape: (batch,)

            # Actual direction: -1 if negative, +1 if positive
            actual_direction = torch.where(actual_returns >= 0,
                                           torch.tensor(1.0, device=device),
                                           torch.tensor(-1.0, device=device))

            # Predicted direction: -1 if negative, +1 if positive
            predicted_direction = torch.where(predicted_returns >= 0,
                                              torch.tensor(1.0, device=device),
                                              torch.tensor(-1.0, device=device))

            # Binary classification direction accuracy
            correct_directions = (predicted_direction == actual_direction).sum().item()
            total_directions = actual_direction.numel()
            direction_accuracy = correct_directions / total_directions

            mse_list.append(mse.item())
            dir_accuracy_list.append(direction_accuracy)

    return (sum(mse_list) / len(mse_list),
            1 - sum(dir_accuracy_list) / len(dir_accuracy_list))

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

num_layers = 2
hidden_size = 32

# Assuming model, train_loader, and val_loader are already defined
saved_models_dir = "/Users/jamesguan/Project/StatsLab/time_series_intro_class/final_project/temp/two_layer_50_hidden_32/"  # Replace with the actual path


# Create a list to store matching epochs
matched_epochs = []

# Define the matching prefix
# matching_prefix = f"{num_layers}_layer_{hidden_size}_size_model_epoch_"
matching_prefix = f"model_epoch_"
train_mse = []
val_mse = []
train_dir_accuracy = []
val_dir_accuracy = []


# Iterate through the files in the directory
for filename in sorted(os.listdir(saved_models_dir)):
    if filename.startswith(matching_prefix) and filename.endswith(".pth"):
        try:
            # Extract the epoch number from the filename
            epoch = int(filename.split('_')[-1].split('.')[0])
            matched_epochs.append(epoch)
        except ValueError:
            print(f"Skipping file {filename}: unable to extract epoch.")

# Sort the epochs in ascending order
matched_epochs.sort()

# Print the matched epochs
print(f"Matched epochs for num_layers={num_layers} and hidden_size={hidden_size}: {matched_epochs}")

model = TwoLayerLSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)

# Iterate through the matched epochs
for epoch in matched_epochs:
    # Load the model weights
    model_path = os.path.join(saved_models_dir, f"{matching_prefix}{epoch}.pth")
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))  # Correctly apply map_location here
    model.load_state_dict(state_dict)  # Load the state_dict into the model

    # Evaluate metrics on training and validation datasets
    train_metrics = evaluate_metrics(model, train_loader)
    val_metrics = evaluate_metrics(model, val_loader)

    # Store the metrics
    train_mse.append(train_metrics[0])
    train_dir_accuracy.append(train_metrics[1])
    val_mse.append(val_metrics[0])
    val_dir_accuracy.append(val_metrics[1])

# Plot MSE
plt.figure(figsize=(12, 6))
plt.plot(matched_epochs, train_mse, label='Training MSE', marker='o')
plt.plot(matched_epochs, val_mse, label='Validation MSE', marker='o')
plt.title(f'MSE Over Epochs for {num_layers} Layers and {hidden_size} Hidden Size')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)

# Save the figure before showing it
mse_plot_filename = f'mse_epochs_{num_layers}_layers_{hidden_size}_hidden_size.png'
plt.savefig(mse_plot_filename, dpi=300)  # Save with high resolution
plt.show()  # Display the figure after saving

# Plot Binary Directional Accuracy
plt.figure(figsize=(12, 6))
plt.plot(matched_epochs, train_dir_accuracy, label='Training Binary Directional Accuracy', marker='o')
plt.plot(matched_epochs, val_dir_accuracy, label='Validation Binary Directional Accuracy', marker='o')
plt.title(f'Binary Directional Accuracy Over Epochs for {num_layers} Layers and {hidden_size} Hidden Size')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Save the figure before showing it
accuracy_plot_filename = f'directional_accuracy_epochs_{num_layers}_layers_{hidden_size}_hidden_size.png'
plt.savefig(accuracy_plot_filename, dpi=300)  # Save with high resolution
plt.show()  # Display the figure after saving


#         # Load the model weights
#         model.load_state_dict(torch.load(os.path.join(saved_models_dir, filename)))
#
#         # Evaluate metrics on training and validation datasets
#         train_metrics = evaluate_metrics(model, train_loader)
#         val_metrics = evaluate_metrics(model, val_loader)
#
#         # Store the metrics
#         train_mse.append(train_metrics[0])
#         train_dir_accuracy.append(train_metrics[1])
#         val_mse.append(val_metrics[0])
#         val_dir_accuracy.append(val_metrics[1])
#
# # Plot MSE
# plt.figure(figsize=(12, 6))
# plt.plot(epochs, train_mse, label='Training MSE', marker='o')
# plt.plot(epochs, val_mse, label='Validation MSE', marker='o')
# plt.title('MSE Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('MSE')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # Plot Binary Directional Accuracy
# plt.figure(figsize=(12, 6))
# plt.plot(epochs, train_dir_accuracy, label='Training Binary Directional Accuracy', marker='o')
# plt.plot(epochs, val_dir_accuracy, label='Validation Binary Directional Accuracy', marker='o')
# plt.title('Binary Directional Accuracy Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()

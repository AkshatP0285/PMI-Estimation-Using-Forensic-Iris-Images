from operator import le
import os
import sys
import numpy as np
import torch
import pandas as pd
from PIL import Image, ImageEnhance
# import tensorflow as tf
from tqdm import tqdm
import torch.nn as nn
import torchvision.models as models
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from torchvision import transforms
import json
from pathlib import Path
import torch.optim as optim
from modules.dataset import CustomDataset
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.set_per_process_memory_fraction(0.50, device= 0)

# Define the paths and other parameters
image_root_dir = r'/DATA2/akshay/Akshat/Warsaw-BioBase-Postmortem-Iris-v3/RGB_Cropped_ISO-Resolution'
metadata_file_path = r'/DATA2/akshay/Akshat/train-test-data-list/RGB/'
output_dir = r'/DATA2/akshay/Akshat/10_fold_results/RGB/'
checkpoint_dir =  '/DATA2/akshay/Akshat/checkpoints/rgb_sub_inception/'
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

def save_checkpoint(state, fold, epoch, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_fold_{fold+1}_epoch_{epoch}.pt')
    torch.save(state, checkpoint_path)
    # Save latest checkpoint reference
    with open(os.path.join(checkpoint_dir, f'latest_fold_{fold+1}.json'), 'w') as f:
        json.dump({'epoch': epoch, 'path': checkpoint_path}, f)

def load_checkpoint(fold, checkpoint_dir):
    latest_file = os.path.join(checkpoint_dir, f'latest_fold_{fold+1}.json')
    if os.path.exists(latest_file):
        with open(latest_file, 'r') as f:
            latest = json.load(f)
        if os.path.exists(latest['path']):
            print(f"Loading checkpoint from epoch {latest['epoch']}")
            return torch.load(latest['path']), latest['epoch']
    return None, 0

# Add this function after the existing checkpoint functions
def cleanup_old_checkpoints(fold, current_epoch, checkpoint_dir, keep_last_n=2):
    """
    Clean up old checkpoints keeping only the last n epochs
    Args:
        fold: Current fold number
        current_epoch: Current epoch number
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
    """
    # Get all checkpoint files for this fold
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith(f'checkpoint_fold_{fold+1}_epoch_') and file.endswith('.pt'):
            checkpoint_files.append(file)
    
    # Sort checkpoints by epoch number
    checkpoint_files.sort(key=lambda x: int(x.split('_epoch_')[1].split('.')[0]))
    
    # Remove old checkpoints if we have more than keep_last_n
    if len(checkpoint_files) > keep_last_n:
        for checkpoint_file in checkpoint_files[:-keep_last_n]:
            file_path = os.path.join(checkpoint_dir, checkpoint_file)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting {file_path}: {e}")


# Model Parameters
model_name = 'inception'
pretrained = False
solver_name = 'Adam'
batch_size = 16
lr = 0.0001
weight_decay = True

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training device: {device} | Pretrained: {pretrained} | Solver Name: {solver_name} | Weight Decay: {weight_decay}')

# Load pre-trained model
if model_name == "vgg":
    model = models.vgg19(pretrained=pretrained)
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, 1)
    model = model.to(device)
elif model_name == "resnet":    
    model = models.resnet152(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model = model.to(device)
elif model_name == "inception":
    model = models.inception_v3(pretrained=pretrained, aux_logits=False)
    model.aux_logits = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    model = model.to(device)
elif model_name == "densenet":
    model = models.densenet121(pretrained=pretrained)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 1)
    model = model.to(device)
else:
    print("Model not found. Exiting.")
    sys.exit()

print(f'Training Network:\n{model}')

# Define loss function and optimizer
criterion = nn.MSELoss()
if solver_name == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6 if weight_decay else 0)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6 if weight_decay else 0)

# Define transformations to apply to the images
if model_name == "inception": 
    trainTransform = transforms.Compose([
        transforms.Resize(299),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(2.0)),
        transforms.ToTensor()
    ])

    testTransform = transforms.Compose([
        transforms.Resize(299),
        transforms.ToTensor()
        ])

else:
    trainTransform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-30, 30)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.Lambda(lambda img: ImageEnhance.Sharpness(img).enhance(2.0)),
    transforms.ToTensor()
    ])

    testTransform = transforms.Compose([
        transforms.ToTensor()
        ])


# Perform 10-fold cross-validation
mse_fold_list = []
rmse_fold_list = []
mae_fold_list = []

for fold in range(0, 10):
    train_data = pd.read_csv(f"{metadata_file_path}fold_{fold+1}_train.csv")
    test_data = pd.read_csv(f"{metadata_file_path}fold_{fold+1}_test.csv")
    val_data = pd.read_csv(f"{metadata_file_path}fold_{fold+1}_validation.csv")

    # Create instances of the CustomDataset
    train_dataset = CustomDataset(data=train_data, root_dir=image_root_dir, transform=trainTransform)
    test_dataset = CustomDataset(data=test_data, root_dir=image_root_dir, transform=testTransform)
    val_dataset = CustomDataset(data=val_data, root_dir=image_root_dir, transform=testTransform)

    # Create DataLoaders for batching and shuffling
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    checkpoint, start_epoch = load_checkpoint(fold, checkpoint_dir)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_mae = checkpoint['best_mae']  
    else:
        start_epoch = 0
        best_mae = float('inf')
    
    best_model_path = f"{output_dir}{model_name}_RGB_opt_{solver_name}_pret_{str(pretrained)}_wd_{str(weight_decay)}_best_model_fold_{fold+1}.pth"
    
    # Training loop
    num_epochs = 300
    patience = 50
    best_mae = float('inf')

    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        total_running_loss = 0.0

        for filenames, inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.unsqueeze(1).to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # del loss
            # torch.cuda.empty_cache(
            total_running_loss += loss.item()


        # To save validation data output
        val_all_file_names = []
        val_all_actual_values = []
        val_all_predicted_values = []

        # Validation
        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            total_mse = 0.0
            total_rmse = 0.0
            total_mae = 0.0

            for filenames, inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.unsqueeze(1).to(device, dtype=torch.float)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

                actual = targets.view(-1).detach().cpu().numpy()
                predicted = outputs.view(-1).detach().cpu().numpy()

                total_mse += mean_squared_error(actual, predicted)
                total_rmse += np.sqrt(mean_squared_error(actual, predicted))
                total_mae += mean_absolute_error(actual, predicted)

                # Append current batch values to the lists
                val_all_file_names.extend(filenames)
                val_all_actual_values.extend(actual)
                val_all_predicted_values.extend(predicted)

                del inputs, targets, outputs, loss
                # torch.cuda.empty_cache()

        train_loss = total_running_loss / len(train_dataloader)
        val_loss = total_val_loss / len(test_dataloader)
        val_mse = total_mse / len(test_dataloader)
        val_rmse = total_rmse / len(test_dataloader)
        val_mae = total_mae / len(test_dataloader)

        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': train_loss,
            'best_mae': best_mae,    
        }
        save_checkpoint(checkpoint_state, fold, epoch+1, checkpoint_dir)
        cleanup_old_checkpoints(fold, epoch+1, checkpoint_dir)

        # Save best model based on MAE
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), best_model_path)

            # Create a DataFrame for actual and predicted values
            val_results = pd.DataFrame({'filename': val_all_file_names, 'Actual': val_all_actual_values, 'Predicted': val_all_predicted_values})
            # Save the results DataFrame to a text file
            val_results.to_csv(f"{output_dir}{model_name}_RGB_val_result_fold_{fold+1}.txt", sep=',', index=False)

            print(f"Fold [{fold+1}] Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.2f} Val Loss: {val_loss:.2f} "
            f"MSE: {val_mse:.2f} RMSE: {val_rmse:.2f} MAE: {val_mae:.2f}")

            
        # torch.cuda.empty_cache()


    # Load the best model
    model.load_state_dict(torch.load(best_model_path))

    # Test your model on new data if needed
    all_file_names = []
    all_actual_values = []
    all_predicted_values = []

    model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        total_mse = 0.0
        total_rmse = 0.0
        total_mae = 0.0

        for filenames, inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.unsqueeze(1).to(device, dtype=torch.float)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_val_loss += loss.item()

            actual = targets.view(-1).detach().cpu().numpy()
            predicted = outputs.view(-1).detach().cpu().numpy()

            total_mse += mean_squared_error(actual, predicted)
            total_rmse += np.sqrt(mean_squared_error(actual, predicted))
            total_mae += mean_absolute_error(actual, predicted)

            # Append current batch values to the lists
            all_file_names.extend(filenames)
            all_actual_values.extend(actual)
            all_predicted_values.extend(predicted)

    # val_loss = total_val_loss / len(test_dataloader)
    val_loss = 0 
    val_mse = total_mse / len(test_dataloader)
    val_rmse = total_rmse / len(test_dataloader)
    val_mae = total_mae / len(test_dataloader)

    mse_fold_list.append(val_mse)
    rmse_fold_list.append(val_rmse)
    mae_fold_list.append(val_mae)

    print('-------------------------------------')
    print('--------Best Model Performance-------')
    print('-------------------------------------')
    print(f'       Fold: {fold+1}')
    print(f'       MSE: {val_mse:.2f}')
    print(f'       RMSE: {val_rmse:.2f}')
    print(f'       MAE: {val_mae:.2f}')

    # Customize matplotlib and seaborn
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['font.size'] = 15
    plt.rcParams["figure.figsize"] = (8, 6)

    # Create a scatter plot with diagonal line
    plt.scatter(all_actual_values, all_predicted_values, color='blue', label='Predicted')
    plt.plot(all_actual_values, all_actual_values, color='red', label='Actual', linestyle='dashed')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}{model_name}_RGB_plot_fold_{fold+1}.pdf", format='pdf', dpi=3000)
    plt.clf()

    # Create a DataFrame for actual and predicted values
    results = pd.DataFrame({'filename': all_file_names, 'Actual': all_actual_values, 'Predicted': all_predicted_values})
    # Save the results DataFrame to a text file
    results.to_csv(f"{output_dir}{model_name}_RGB_test_result_fold_{fold+1}.txt", sep=',', index=False)


print('------------------------------------------')
print('-------- Model Average Performance -------')
print('------------------------------------------')
print(f'        Avg MSE: {(sum(mse_fold_list)/len(mse_fold_list)):.2f}') 
print(f'       Avg RMSE: {(sum(rmse_fold_list)/len(rmse_fold_list)):.2f}')
print(f'        Avg MAE: {(sum(mae_fold_list)/len(mae_fold_list)):.2f}')
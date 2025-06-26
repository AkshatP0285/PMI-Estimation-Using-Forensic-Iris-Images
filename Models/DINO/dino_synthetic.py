import torch
import timm
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib

# =================== CONFIG ===================
data_root = "/DATA2/akshay/Akshat/UND-SFI"  # Replace with your actual data root path
metadata_path = "/DATA2/akshay/Akshat/metadata/synthetic_random_pmi.csv"  # Replace with your actual metadata path
model_save_path = "mlp_model.joblib"
scaler_save_path = "scaler.joblib"
# ==============================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load pretrained DINO model
model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
model.eval()
model.to(device)

# Load metadata
metadata = pd.read_csv(metadata_path)
filename_to_pmi = dict(zip(metadata['filename'], metadata['pmi']))

# Collect image paths and corresponding PMI values
image_paths = []
pmi_values = []

for subfolder in sorted(os.listdir(data_root)):
    folder_path = os.path.join(data_root, subfolder)
    if os.path.isdir(folder_path) and subfolder.startswith("class_"):
        for fname in os.listdir(folder_path):
            if fname.endswith(".bmp"):
                full_path = os.path.join(folder_path, fname)
                if fname in filename_to_pmi:
                    image_paths.append(full_path)
                    pmi_values.append(filename_to_pmi[fname])
                else:
                    print(f"Warning: {fname} not found in metadata!")

# Function to extract DINO features
def extract_dino_features(image_paths, batch_size=32):
    features = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting DINO Features"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        for path in batch_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Unable to load image {path}")
                batch_images.append(torch.zeros((3, 224, 224)))
                continue

            img = cv2.resize(img, (224, 224))
            img = np.stack([img] * 3, axis=-1)
            img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
            batch_images.append(img)

        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            batch_features = model.forward_features(batch_tensor)
            batch_features = batch_features.cpu().numpy()
        features.append(batch_features)

    return np.vstack(features)

# Extract features
dino_features = extract_dino_features(image_paths)
dino_features = dino_features[:, 0, :]  # CLS token

# Train-test split and standardization
X = pd.DataFrame(dino_features)
y = np.array(pmi_values)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train MLP Regressor
mlp = MLPRegressor(
    hidden_layer_sizes=(512, 216, 128),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=True,
    learning_rate='adaptive',
    learning_rate_init=0.001
)

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)

# Evaluation
print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")

# Save model and scaler
joblib.dump(mlp, model_save_path)
joblib.dump(scaler, scaler_save_path)
print(f"Model saved to {model_save_path}")
print(f"Scaler saved to {scaler_save_path}")

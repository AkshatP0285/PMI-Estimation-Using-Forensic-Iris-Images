import torch
import timm
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib

from torchvision import transforms
from PIL import Image

# =================== CONFIG ===================
nir_dir = "/DATA2/akshay/Akshat/Warsaw-BioBase-Postmortem-Iris-v3/NIR_ISO-Resolution"
rgb_dir = "/DATA2/akshay/Akshat/Warsaw-BioBase-Postmortem-Iris-v3/RGB_Cropped_ISO-Resolution"
folds_dir = "/DATA2/akshay/Akshat/train-test-data-list/multispectral"
embedding_cache_path = "warsaw_dino_embeddings_multispectral.npz"
device = "cuda" if torch.cuda.is_available() else "cpu"
# ==============================================

print(f"Device: {device}")

# Load pretrained DINO model
model = timm.create_model("vit_base_patch16_224_dino", pretrained=True)
model.eval().to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------- STEP 1: Precompute or Load DINO Embeddings ----------
def extract_multispectral_features(df, nir_dir, rgb_dir, batch_size=32):
    features = []
    filenames = []

    for i in tqdm(range(0, len(df), batch_size), desc="Extracting MultiSpectral DINO Features"):
        batch_nir = []
        batch_rgb = []

        for _, row in df.iloc[i:i+batch_size].iterrows():
            nir_path = os.path.join(nir_dir, row['nir_filename'])
            rgb_path = os.path.join(rgb_dir, row['rgb_filename'])

            try:
                nir_img = Image.open(nir_path).convert('L')
                rgb_img = Image.open(rgb_path).convert('RGB')
                nir_tensor = transform(nir_img).repeat(3, 1, 1)  # Convert 1 channel to 3
                rgb_tensor = transform(rgb_img)
                combined_tensor = (nir_tensor + rgb_tensor) / 2
            except:
                combined_tensor = torch.zeros((3, 224, 224))

            batch_nir.append(combined_tensor)
            filenames.append(row['nir_filename'])

        batch_tensor = torch.stack(batch_nir).to(device)
        with torch.no_grad():
            batch_features = model.forward_features(batch_tensor)
            batch_features = batch_features.cpu().numpy()
        features.append(batch_features)

    return np.vstack(features), filenames

if os.path.exists(embedding_cache_path):
    print("Loading precomputed DINO embeddings...")
    data = np.load(embedding_cache_path, allow_pickle=True)
    all_features = data["features"]
    all_filenames = data["filenames"]
    feature_map = dict(zip(all_filenames, all_features))
else:
    print("Extracting and saving DINO embeddings for all images...")
    all_folds = [os.path.join(folds_dir, f) for f in os.listdir(folds_dir) if f.endswith('.csv')]
    df_all = pd.concat([pd.read_csv(f) for f in all_folds])
    raw_features, all_filenames = extract_multispectral_features(df_all, nir_dir, rgb_dir)
    all_features = raw_features[:, 0, :]  # CLS token
    np.savez(embedding_cache_path, features=all_features, filenames=all_filenames)
    feature_map = dict(zip(all_filenames, all_features))

# ---------- STEP 2: Helper to Load Features for Fold ----------
def get_fold_features_and_labels(csv_path, feature_map):
    df = pd.read_csv(csv_path)
    features, labels = [], []
    for _, row in df.iterrows():
        key = row['nir_filename']
        if key in feature_map:
            features.append(feature_map[key])
            labels.append(row['pmi'])
        else:
            print(f"Warning: {key} not in feature map!")
    return np.array(features), np.array(labels)

mae_scores = []
rmse_scores = []
r2_scores = []

# ---------- STEP 3: 10-Fold Cross Validation ----------
for fold in range(1, 11):
    print(f"\n==================== Fold {fold} ====================")

    train_csv = os.path.join(folds_dir, f"fold_{fold}_train.csv")
    val_csv = os.path.join(folds_dir, f"fold_{fold}_validation.csv")
    test_csv = os.path.join(folds_dir, f"fold_{fold}_test.csv")

    X_train, y_train = get_fold_features_and_labels(train_csv, feature_map)
    X_val, y_val = get_fold_features_and_labels(val_csv, feature_map)
    X_test, y_test = get_fold_features_and_labels(test_csv, feature_map)

    # Combine train and val for final training
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)

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

    mlp.fit(X_train_scaled, y_train_full)
    y_pred = mlp.predict(X_test_scaled)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²: {r2:.3f}")

    mae_scores.append(mae)
    rmse_scores.append(rmse)
    r2_scores.append(r2)

    # Save model and scaler per fold
    joblib.dump(mlp, f"mlp_model_multispectral_fold{fold}.joblib")
    joblib.dump(scaler, f"scaler_multispectral_fold{fold}.joblib")

print("\n==================== Average Performance Across Folds ====================")
print(f"Average MAE : {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
print(f"Average RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")
print(f"Average R²  : {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
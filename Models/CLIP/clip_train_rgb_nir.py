import torch
import open_clip
import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib

# =================== CONFIG ===================
data_root = "/DATA2/akshay/Akshat/Warsaw-BioBase-Postmortem-Iris-v3/RGB_Cropped_ISO-Resolution"
folds_dir = "/DATA2/akshay/Akshat/train-test-data-list/RGB"
embedding_cache_path = "warsaw_openclip_embeddings_rgb.npz"
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B-32"
pretrained = "openai"
# ==============================================

print(f"Device: {device}")

# Load pretrained OpenCLIP model (only vision encoder used)
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
model.eval().to(device)

# ---------- STEP 1: Precompute or Load OpenCLIP Embeddings ----------
def extract_openclip_features(image_paths, batch_size=32):
    features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting OpenCLIP Features"):
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
            img = preprocess(Image.fromarray(img))  # Use OpenCLIP’s preprocessing
            batch_images.append(img)

        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(batch_tensor)
            batch_features = batch_features.cpu().numpy()
        features.append(batch_features)

    return np.vstack(features)

if os.path.exists(embedding_cache_path):
    print("Loading precomputed OpenCLIP embeddings...")
    data = np.load(embedding_cache_path, allow_pickle=True)
    all_features = data["features"]
    all_filenames = data["filenames"]
    feature_map = dict(zip(all_filenames, all_features))
else:
    from PIL import Image  # only needed here if not imported above

    print("Extracting and saving OpenCLIP embeddings for all images...")
    all_folds = [os.path.join(folds_dir, f) for f in os.listdir(folds_dir) if f.endswith('.csv')]
    df_all = pd.concat([pd.read_csv(f) for f in all_folds])
    image_paths = [os.path.join(data_root, fname) for fname in df_all['filename']]

    raw_features = extract_openclip_features(image_paths)
    all_features = raw_features  # no CLS token separation needed
    all_filenames = [os.path.basename(p) for p in image_paths]
    np.savez(embedding_cache_path, features=all_features, filenames=all_filenames)
    feature_map = dict(zip(all_filenames, all_features))

# ---------- STEP 2: Helper to Load Features for Fold ----------
def get_fold_features_and_labels(csv_path, feature_map):
    df = pd.read_csv(csv_path)
    features, labels = [], []
    for _, row in df.iterrows():
        base_fname = os.path.basename(row["filename"])
        if base_fname in feature_map:
            features.append(feature_map[base_fname])
            labels.append(row["pmi"])
        else:
            print(f"Warning: {base_fname} not in feature map!")
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
    joblib.dump(mlp, f"mlp_openclip_model_fold{fold}.joblib")
    joblib.dump(scaler, f"scaler_openclip_fold{fold}.joblib")

print("\n==================== Average Performance Across Folds ====================")
print(f"Average MAE : {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
print(f"Average RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")
print(f"Average R²  : {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")

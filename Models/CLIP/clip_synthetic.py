import torch
import clip
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
from torchvision import transforms

# =================== CONFIG ===================
data_root = "/DATA2/akshay/Akshat/UND-SFI"
metadata_path = "/DATA2/akshay/Akshat/metadata/synthetic_random_pmi.csv"
model_save_path = "clip_mlp_model.joblib"
scaler_save_path = "clip_scaler.joblib"
embedding_save_path = "clip_embeddings.npz"
# ==============================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model.eval()

# Fast preprocessing (mimics CLIP's default but faster)
preprocess_fast = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4815, 0.4578, 0.4082),
                         std=(0.2686, 0.2613, 0.2758))
])

# ================== LOAD METADATA ==================
metadata = pd.read_csv(metadata_path)
filename_to_pmi = dict(zip(metadata['filename'], metadata['pmi']))

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

# ============== CLIP FEATURE EXTRACTION ===============
def extract_clip_features(image_paths, batch_size=32):
    all_features = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting CLIP Features"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        for path in batch_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Failed to load image {path}. Using zeros.")
                img_tensor = torch.zeros(3, 224, 224)
            else:
                img = cv2.resize(img, (224, 224))
                img = np.stack([img]*3, axis=-1)  # Convert to 3 channels
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = preprocess_fast(img)

            batch_images.append(img_tensor)

        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(batch_tensor)
            image_features = image_features.cpu().numpy()

        all_features.append(image_features)

    return np.vstack(all_features)

# ================== EMBEDDING STEP ==================
if os.path.exists(embedding_save_path):
    print(f"Loading saved embeddings from {embedding_save_path}...")
    data = np.load(embedding_save_path)
    clip_features = data["features"]
    pmi_values = data["pmi"]
else:
    print("No saved embeddings found. Extracting CLIP features...")
    raw_features = extract_clip_features(image_paths, batch_size=64)
    clip_features = raw_features
    pmi_values = np.array(pmi_values)
    np.savez(embedding_save_path, features=clip_features, pmi=pmi_values)
    print(f"Embeddings saved to {embedding_save_path}")
# ====================================================

# ======== REGRESSION TRAINING (MLP) ================
X = pd.DataFrame(clip_features)
y = np.array(pmi_values)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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

# ============== EVALUATION ========================
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")

# ============== SAVE MODEL ========================
joblib.dump(mlp, model_save_path)
joblib.dump(scaler, scaler_save_path)
print(f"Model saved to {model_save_path}")
print(f"Scaler saved to {scaler_save_path}")

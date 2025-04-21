import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ----------------------------
# 1. Load and prepare data
# ----------------------------
data = []
with open('V2labels_fixed_with_features.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
df = pd.DataFrame(data)
print("Initial data loaded:", df.shape)

print("Kolommen in df:", df.columns.tolist())

# Use the flattened feature vector directly
# Feature extraction
df['feature_vector'] = df['features']  # ← zorg dat deze als eerste komt
df['vector_length'] = df['feature_vector'].apply(len)
print(df['vector_length'].value_counts())

features_df = pd.DataFrame(df['feature_vector'].tolist())
df = pd.concat([df, features_df], axis=1)

# Verwijder alles wat we niet willen trainen op
df.drop(columns=['features', 'feature_vector', 'vector_length', 'filename', 'crown', 'row', 'col'], inplace=True)



# Map tile types to numeric codes
tile_mapping = {
    "MINE": 0,
    "HOME": 1,
    "TABLE": 2,
    "WHEAT": 3,
    "FOREST": 4,
    "GRASSLAND": 5,
    "SWAMP": 6,
    "LAKE": 7
}
df['tile_type_code'] = df['tile_type'].map(tile_mapping)
df.drop('tile_type', axis=1, inplace=True)

# ----------------------------
# 2. Train/test preparation
# ----------------------------
numeric_df = df.select_dtypes(include='number')
feature_cols = [col for col in numeric_df.columns if col != 'tile_type_code']
X = df[feature_cols].values
y = df['tile_type_code'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_scaled)

# ----------------------------
# 3. Optional: PCA visualization
# ----------------------------
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
plt.figure(figsize=(10, 8))
sns.heatmap(pca_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("PCA Component Correlation Heatmap")
plt.tight_layout()
plt.show()

# ----------------------------
# 4. Train model
# ----------------------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_pca, y)

# ----------------------------
# 5. Evaluate
# ----------------------------
y_pred = clf.predict(X_pca)
print("\nClassification Report:\n", classification_report(y, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("Accuracy:", accuracy_score(y, y_pred))

# ----------------------------
# 6. Save artifacts
# ----------------------------
joblib.dump(clf, 'trained_RF_model.joblib')
joblib.dump(scaler, 'trained_scaler.joblib')
joblib.dump(pca, 'trained_pca.joblib')
with open('feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)

print("\nModel, scaler, PCA and feature structure saved successfully.")

import json
with open("feature_columns.json") as f:
    cols = json.load(f)
print("Length of feature_columns:", len(cols))  # ← MOET 5381 zijn

